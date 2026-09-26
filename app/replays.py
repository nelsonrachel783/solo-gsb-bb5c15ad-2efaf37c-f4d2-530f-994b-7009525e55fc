"""回放录制规划核心逻辑（纯标准库实现，不依赖 Web 框架）。

规则概述
--------
* 现有口令沿用 ``POST /compile`` 的口令数组，先用同一个编排器解析并校验每条口令的
  起止毫秒；候选口令由 ``id`` + 非负整数 ``priority`` 给出，至多 12 条。
* 每个候选生成一个**全局录制窗口**（不分通道）::

      [max(0, start_ms - pre_pad_ms), end_ms + post_pad_ms)

* 入选集合的全部窗口中，**跨通道重叠或首尾相接**的窗口先合并（半开区间，
  端点相等即相接），合并后各片段长度之和才计入预算。
* 目标依次为：① 合并后总时长不超过 ``budget_ms`` 的集合中，候选 ``priority``
  总和最大；② 并列时入选条数最多；③ 再并列时「入选 id 升序列表」字典序最小。
  空集合始终允许（合并片段与总时长分别为空列表与 0）。

实现要点
--------
候选至多 12 条，位掩码枚举全部 2^n 个子集（n=12 时仅 4096 个）；先用低位递推
求出每个子集的条数与 priority 总和，再收集该子集的窗口按起点排序合并，
求合并片段与总时长，按 ``(priority 总和, 条数, 入选 id 升序列表)`` 裁决。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .compiler import CompileError, compile_commands

# 一次回放规划最多允许的候选口令条数
MAX_REPLAY_CANDIDATES = 12


class ReplayError(CompileError):
    """回放录制规划失败的业务异常；复用 CompileError 的 422 转换通道。"""


def plan_replays(
    commands: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    pre_pad_ms: int,
    post_pad_ms: int,
    budget_ms: int,
) -> dict[str, Any]:
    """求合并录制时长不超过预算、且 (priority 总和, 条数, id 列表) 最优的入选集合。

    不会修改传入的 ``commands`` / ``candidates``。

    :raises CompileError: 口令编排非法（重复 id、依赖环、负起点等，与 /compile 相同）。
    :raises ReplayError: 留白 / 预算参数非法、候选非法（空 id、重复 id、超过 12 条、
        负 priority、引用不存在的口令 id）时抛出，``code`` 可用于程序化区分错误类型。
    """
    # 1) 复用编排器解析并校验现有口令（深拷贝，setdefault 等不会回写调用方数据；
    #    回放规划只需解析后的时间线，跳过 O(n²) 的冲突对枚举）
    compiled = compile_commands(deepcopy(commands), with_conflicts=False)
    by_id = {entry["id"]: entry for entry in compiled["timeline"]}

    # 2) 留白与预算参数校验
    if pre_pad_ms < 0 or post_pad_ms < 0:
        raise ReplayError(
            "INVALID_REQUEST",
            "pre_pad_ms 与 post_pad_ms 都必须为非负整数",
            {"pre_pad_ms": pre_pad_ms, "post_pad_ms": post_pad_ms},
        )
    if budget_ms <= 0:
        raise ReplayError(
            "INVALID_REPLAY_BUDGET",
            "budget_ms 必须为正数",
            {"budget_ms": budget_ms},
        )

    # 3) 候选校验，得到按 id 升序排列的候选（id 与 priority）
    items = _validate_candidates(candidates, by_id)

    n = len(items)
    size = 1 << n
    priorities = [item["priority"] for item in items]
    windows = [
        (
            max(0, by_id[item["id"]]["start_ms"] - pre_pad_ms),
            by_id[item["id"]]["end_ms"] + post_pad_ms,
        )
        for item in items
    ]

    # 4) 低位递推：每个子集的条数与 priority 总和
    popcount = [0] * size
    sub_pri = [0] * size
    for mask in range(1, size):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        popcount[mask] = popcount[rest] + 1
        sub_pri[mask] = sub_pri[rest] + priorities[i]

    # 5) 枚举子集：收集窗口 -> 跨通道合并 -> 总时长；预算内按
    #    (priority 总和最大, 条数最多, 入选 id 升序列表字典序最小) 裁决。
    #    空集合总时长 0，必然可行，故恒存在最优解。
    best_mask = 0
    best_segments: list[tuple[int, int]] = []
    best_total = 0
    best_key: tuple[int, int, tuple[str, ...]] = (
        0,
        0,
        (),
    )  # (priority 总和, 条数, id 升序元组)
    for mask in range(size):
        selected = [
            windows[i] for i in range(n) if mask >> i & 1
        ]
        segments = merge_windows(selected)
        total = sum(end - start for start, end in segments)
        if total > budget_ms:
            continue
        ids_tuple = tuple(items[i]["id"] for i in range(n) if mask >> i & 1)
        key = (sub_pri[mask], popcount[mask], ids_tuple)
        if (
            key[0] > best_key[0]
            or (
                key[0] == best_key[0]
                and (
                    key[1] > best_key[1]
                    or (key[1] == best_key[1] and key[2] < best_key[2])
                )
            )
        ):
            best_key = key
            best_mask = mask
            best_segments = segments
            best_total = total

    return {
        "selected_ids": [
            items[i]["id"] for i in range(n) if best_mask >> i & 1
        ],
        "segments": [
            {"start_ms": start, "end_ms": end} for start, end in best_segments
        ],
        "total_duration_ms": best_total,
    }


def merge_windows(
    windows: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """把半开窗口按起点排序后求并集：重叠或首尾相接（end 等于下一个 start）即合并。

    合并不区分窗口来自哪条通道——这正是“跨通道重叠或首尾相接也合并”的全局窗口。
    """
    if not windows:
        return []
    ordered = sorted(windows)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _validate_candidates(
    candidates: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """校验候选列表，返回按 id 升序排列的候选 dict 列表。"""
    # 1) 空 id
    empty_ids = [cand["id"] for cand in candidates if cand["id"] == ""]
    if empty_ids:
        raise ReplayError(
            "INVALID_ID", "候选口令 id 不能为空", {"ids": empty_ids}
        )

    # 2) 重复 id
    seen: set[str] = set()
    duplicates: list[str] = []
    for cand in candidates:
        cid = cand["id"]
        if cid in seen and cid not in duplicates:
            duplicates.append(cid)
        seen.add(cid)
    if duplicates:
        raise ReplayError(
            "DUPLICATE_ID",
            f"存在重复的候选口令 id: {', '.join(sorted(duplicates))}",
            {"duplicates": sorted(duplicates)},
        )

    # 3) 条数上限
    if len(candidates) > MAX_REPLAY_CANDIDATES:
        raise ReplayError(
            "TOO_MANY_REPLAY_CANDIDATES",
            f"最多 {MAX_REPLAY_CANDIDATES} 条候选口令参与一次回放规划，"
            f"当前 {len(candidates)} 条",
            {"count": len(candidates), "limit": MAX_REPLAY_CANDIDATES},
        )

    # 4) priority 必须为非负整数
    bad_priority = sorted(
        cand["id"] for cand in candidates if cand["priority"] < 0
    )
    if bad_priority:
        raise ReplayError(
            "INVALID_PRIORITY",
            "候选 priority 必须为非负整数",
            {"ids": bad_priority},
        )

    # 5) 候选口令必须存在
    unknown = sorted(cand["id"] for cand in candidates if cand["id"] not in by_id)
    if unknown:
        raise ReplayError(
            "REPLAY_TARGET_NOT_FOUND",
            f"候选引用了不存在的口令 id: {', '.join(unknown)}",
            {"ids": unknown},
        )

    return sorted(candidates, key=lambda cand: cand["id"])
