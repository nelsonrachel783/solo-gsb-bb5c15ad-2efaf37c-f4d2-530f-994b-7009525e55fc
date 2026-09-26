"""app.replays 的单元测试（仅依赖标准库，可直接 `python -m unittest discover`）。"""

from __future__ import annotations

import random
import unittest
from itertools import combinations

from app.compiler import START, CompileError
from app.replays import MAX_REPLAY_CANDIDATES, merge_windows, plan_replays


def cmd(cid, channel="m", anchor=START, offset=0, duration=1000):
    return {
        "id": cid,
        "channel": channel,
        "anchor": anchor,
        "offset_ms": offset,
        "duration_ms": duration,
    }


def candidate(cid, priority=0):
    return {"id": cid, "priority": priority}


def windows_of(commands, ids, pre, post):
    """测试侧独立计算候选窗口（与核心逻辑分开书写）。"""
    from app.compiler import compile_commands

    by_id = {e["id"]: e for e in compile_commands([dict(c) for c in commands])["timeline"]}
    return [
        (max(0, by_id[cid]["start_ms"] - pre), by_id[cid]["end_ms"] + post)
        for cid in ids
    ]


def merge_sorted(windows):
    """独立的排序合并实现，用于交叉验证 merge_windows。"""
    ordered = sorted(windows)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def brute_force(commands, candidates, pre, post, budget):
    """枚举全部子集，按题面三目标求最优（与核心逻辑分开书写）。"""
    by_id = {c["id"]: c for c in candidates}
    all_ids = sorted(by_id)

    best_ids: list[str] = []
    best_key = (-1, -1, ())  # priority 总和、条数取最大；id 元组取最小
    best_segments: list[tuple[int, int]] = []
    best_total = -1
    for r in range(len(all_ids) + 1):
        for chosen in combinations(all_ids, r):
            wins = windows_of(commands, chosen, pre, post)
            segments = merge_sorted(wins)
            total = sum(e - s for s, e in segments)
            if total > budget:
                continue
            pri = sum(by_id[cid]["priority"] for cid in chosen)
            key = (pri, len(chosen), tuple(chosen))
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
                best_ids = list(chosen)
                best_segments = segments
                best_total = total

    return {
        "selected_ids": best_ids,
        "segments": [{"start_ms": s, "end_ms": e} for s, e in best_segments],
        "total_duration_ms": best_total,
    }


class MergeWindowTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_windows([]), [])

    def test_disjoint_kept_separate(self):
        self.assertEqual(merge_windows([(0, 10), (20, 30)]), [(0, 10), (20, 30)])

    def test_touching_merges(self):
        # 半开区间首尾相接（end == 下一个 start）即合并
        self.assertEqual(merge_windows([(0, 10), (10, 20)]), [(0, 20)])

    def test_overlap_merges(self):
        self.assertEqual(merge_windows([(0, 10), (5, 15)]), [(0, 15)])

    def test_unsorted_and_contained(self):
        self.assertEqual(
            merge_windows([(30, 40), (0, 50), (5, 10)]), [(0, 50)]
        )

    def test_chain_touch(self):
        self.assertEqual(
            merge_windows([(0, 10), (10, 20), (20, 30)]), [(0, 30)]
        )


class BasicPlanTests(unittest.TestCase):
    def setUp(self):
        # a [0,1000)；b [2000,2600)；c [2700,3300)（b、c 窗口加留白后首尾相接）；
        # d（sub 通道）[3800,4200)
        self.commands = [
            cmd("a", "main", START, 0, 1000),
            cmd("b", "main", START, 2000, 600),
            cmd("c", "main", START, 2700, 600),
            cmd("d", "sub", START, 3800, 400),
        ]

    def test_empty_candidates(self):
        result = plan_replays(self.commands, [], 0, 0, 1)
        self.assertEqual(result, {"selected_ids": [], "segments": [], "total_duration_ms": 0})

    def test_single_candidate_window_with_pads(self):
        # a [0,1000)，前留白 100 被 0 截断，后留白 200 -> [0,1200)
        result = plan_replays(self.commands, [candidate("a")], 100, 200, 5000)
        self.assertEqual(
            result,
            {
                "selected_ids": ["a"],
                "segments": [{"start_ms": 0, "end_ms": 1200}],
                "total_duration_ms": 1200,
            },
        )

    def test_pre_pad_clamped_at_zero_from_negative_offset_case(self):
        # 口令起点 100，前留白 300 -> 窗口左端 max(0, -200) = 0
        commands = [cmd("x", offset=100, duration=50)]
        result = plan_replays(commands, [candidate("x")], 300, 0, 1000)
        self.assertEqual(
            result["segments"], [{"start_ms": 0, "end_ms": 150}]
        )

    def test_touching_across_channels_merge(self):
        # 前/后留白各 100：
        # b -> [1900,2700)，c -> [2600,3400)（重叠合并），d(sub) -> [3700,4300)
        # b+c 合并片段 [1900,3400) 长 1500；d 与它相距 300 不合并
        result = plan_replays(
            self.commands,
            [candidate("b"), candidate("c"), candidate("d")],
            100,
            100,
            5000,
        )
        self.assertEqual(
            result["segments"],
            [
                {"start_ms": 1900, "end_ms": 3400},
                {"start_ms": 3700, "end_ms": 4300},
            ],
        )
        self.assertEqual(result["total_duration_ms"], 2100)

    def test_exactly_end_equals_next_start_merges(self):
        # 无留白时 b [2000,2600)、c [2700,3300) 不相接；后留白 100 让两个窗口
        # 变为 b [2000,2700)、c [2700,3400)，首尾相接合并为 [2000,3400)
        result = plan_replays(
            self.commands,
            [candidate("b"), candidate("c")],
            0,
            100,
            5000,
        )
        self.assertEqual(
            result["segments"], [{"start_ms": 2000, "end_ms": 3400}]
        )
        self.assertEqual(result["total_duration_ms"], 1400)

    def test_budget_selects_max_priority_sum(self):
        # 无留白窗口：a 长 1000，b 长 600，c 长 600，d 长 400；b、c 相距 100 不合并。
        # 预算 1200：{b,c} 时长 1200、priority 10；{a} 1000、5；{b,d} 1000、9
        cands = [
            candidate("a", 5),
            candidate("b", 8),
            candidate("c", 2),
            candidate("d", 1),
        ]
        result = plan_replays(self.commands, cands, 0, 0, 1200)
        self.assertEqual(result["selected_ids"], ["b", "c"])
        self.assertEqual(
            result["segments"],
            [{"start_ms": 2000, "end_ms": 2600}, {"start_ms": 2700, "end_ms": 3300}],
        )
        self.assertEqual(result["total_duration_ms"], 1200)

    def test_merging_makes_combination_cheaper_than_sum(self):
        # 前/后留白各 100：b -> [1900,2700)、c -> [2600,3400)，重叠 100ms，
        # 合并为 [1900,3400) 长 1500，比两窗口长度之和 800+800=1600 省 100。
        cands = [candidate("b", 1), candidate("c", 1)]
        result = plan_replays(self.commands, cands, 100, 100, 1500)
        self.assertEqual(result["selected_ids"], ["b", "c"])
        self.assertEqual(
            result["segments"], [{"start_ms": 1900, "end_ms": 3400}]
        )
        self.assertEqual(result["total_duration_ms"], 1500)
        # 预算 1499 时无法同时选入（合计按合并计 1500）；单条窗口各长 800，
        # priority 相同只能取 1 条，id 小者 b 胜出
        too_tight = plan_replays(self.commands, cands, 100, 100, 1499)
        self.assertEqual(too_tight["selected_ids"], ["b"])

    def test_tie_break_more_count(self):
        # 预算 1000：a 单条长 1000、priority 5；b+c 两条各 600（不相交，合计 1200
        # 超预算）-> 只能单条；改为预算 1200：{b,c} 合计 1200、priority 3+0=3，
        # {a} priority 5 更高仍取 a。这里构造 priority 相同比条数：
        # a pri 3；b pri 2、c pri 1（合计 pri 3、2 条）-> 取 b,c。
        cands = [candidate("a", 3), candidate("b", 2), candidate("c", 1)]
        result = plan_replays(self.commands, cands, 0, 0, 1200)
        self.assertEqual(result["selected_ids"], ["b", "c"])

    def test_tie_break_lexicographic_ids(self):
        # b、d 窗口不相交、各长 600/400；预算只允许 1 条（如 700），
        # 两者 priority 相同，取条数相同（1 条），id 升序列表字典序小者 b
        cands = [candidate("d", 7), candidate("b", 7)]
        result = plan_replays(self.commands, cands, 0, 0, 700)
        self.assertEqual(result["selected_ids"], ["b"])

    def test_zero_budget_only_empty_set(self):
        # 所有窗口长度为正：预算为 0 在核心层非法；预算极小（1）时只能空集
        result = plan_replays(
            self.commands, [candidate("a", 9)], 0, 0, 1
        )
        self.assertEqual(result["selected_ids"], [])
        self.assertEqual(result["total_duration_ms"], 0)
        self.assertEqual(result["segments"], [])

    def test_zero_length_command_window_still_positive(self):
        # duration 必为正，故候选窗口长度恒正；零留白也不会出现零长窗口
        commands = [cmd("z", duration=1)]
        result = plan_replays(commands, [candidate("z")], 0, 0, 1)
        self.assertEqual(
            result,
            {
                "selected_ids": ["z"],
                "segments": [{"start_ms": 0, "end_ms": 1}],
                "total_duration_ms": 1,
            },
        )


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.commands = [cmd("a"), cmd("b")]

    def test_unknown_id(self):
        with self.assertRaises(CompileError) as ctx:
            plan_replays(self.commands, [candidate("ghost")], 0, 0, 10)
        self.assertEqual(ctx.exception.code, "REPLAY_TARGET_NOT_FOUND")
        self.assertEqual(ctx.exception.details, {"ids": ["ghost"]})

    def test_unknown_ids_sorted(self):
        with self.assertRaises(CompileError) as ctx:
            plan_replays(
                self.commands, [candidate("z"), candidate("m"), candidate("q")],
                0, 0, 10,
            )
        self.assertEqual(ctx.exception.code, "REPLAY_TARGET_NOT_FOUND")
        self.assertEqual(ctx.exception.details["ids"], ["m", "q", "z"])

    def test_empty_id(self):
        with self.assertRaises(CompileError) as ctx:
            plan_replays(self.commands, [candidate("")], 0, 0, 10)
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_duplicate_id(self):
        with self.assertRaises(CompileError) as ctx:
            plan_replays(
                self.commands, [candidate("a"), candidate("a")], 0, 0, 10
            )
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")
        self.assertEqual(ctx.exception.details, {"duplicates": ["a"]})

    def test_negative_priority(self):
        with self.assertRaises(CompileError) as ctx:
            plan_replays(self.commands, [candidate("a", -1)], 0, 0, 10)
        self.assertEqual(ctx.exception.code, "INVALID_PRIORITY")
        self.assertEqual(ctx.exception.details, {"ids": ["a"]})

    def test_too_many_candidates(self):
        commands = [cmd(f"c{i:02d}") for i in range(MAX_REPLAY_CANDIDATES + 1)]
        cands = [candidate(f"c{i:02d}") for i in range(MAX_REPLAY_CANDIDATES + 1)]
        with self.assertRaises(CompileError) as ctx:
            plan_replays(commands, cands, 0, 0, 10**12)
        self.assertEqual(ctx.exception.code, "TOO_MANY_REPLAY_CANDIDATES")
        self.assertEqual(
            ctx.exception.details,
            {"count": 13, "limit": MAX_REPLAY_CANDIDATES},
        )

    def test_negative_pads(self):
        with self.assertRaises(CompileError) as ctx:
            plan_replays(self.commands, [candidate("a")], -1, 0, 10)
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")
        with self.assertRaises(CompileError) as ctx:
            plan_replays(self.commands, [candidate("a")], 0, -1, 10)
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_non_positive_budget(self):
        for budget in (0, -5):
            with self.assertRaises(CompileError) as ctx:
                plan_replays(self.commands, [candidate("a")], 0, 0, budget)
            self.assertEqual(ctx.exception.code, "INVALID_REPLAY_BUDGET")

    def test_compile_error_takes_precedence(self):
        # commands 编排非法（缺失锚点）时优先返回编排错误，即便候选也非法
        bad_commands = [cmd("a", anchor="ghost")]
        with self.assertRaises(CompileError) as ctx:
            plan_replays(bad_commands, [candidate("nope")], 0, 0, 10)
        self.assertEqual(ctx.exception.code, "MISSING_ANCHOR")

    def test_does_not_mutate_inputs(self):
        commands = [cmd("a"), {"id": "b", "channel": "m", "duration_ms": 10}]
        cands = [candidate("b")]
        plan_replays(commands, cands, 0, 0, 1000)
        self.assertNotIn("anchor", commands[1])
        self.assertNotIn("offset_ms", commands[1])
        self.assertEqual(cands[0], {"id": "b", "priority": 0})


class FuzzTests(unittest.TestCase):
    """随机实例下与独立暴力枚举逐字段对照。"""

    def test_random_cases_against_brute_force(self):
        rng = random.Random(20260926)
        for _ in range(400):
            n = rng.randrange(0, 8)
            commands = []
            for i in range(n):
                anchor = START
                if i > 0 and rng.random() < 0.5:
                    anchor = commands[rng.randrange(i)]["id"]
                commands.append(
                    cmd(
                        f"c{i}",
                        channel=rng.choice(["m", "s", "x"]),
                        anchor=anchor,
                        offset=rng.randrange(0, 50),
                        duration=rng.randrange(1, 80),
                    )
                )
            k = rng.randrange(0, n + 1)
            ids = rng.sample([f"c{i}" for i in range(n)], k) if n else []
            cands = [candidate(cid, rng.randrange(0, 6)) for cid in ids]
            pre = rng.randrange(0, 40)
            post = rng.randrange(0, 40)
            budget = rng.randrange(0, 300)

            result = plan_replays(
                [dict(c) for c in commands],
                [dict(c) for c in cands],
                pre,
                post,
                budget,
            )
            expected = brute_force(commands, cands, pre, post, budget)
            self.assertEqual(result["selected_ids"], expected["selected_ids"])
            self.assertEqual(result["segments"], expected["segments"])
            self.assertEqual(result["total_duration_ms"], expected["total_duration_ms"])
            # 合并片段必须按起点升序、互不重叠且不相接
            starts = [s["start_ms"] for s in result["segments"]]
            self.assertEqual(starts, sorted(starts))
            for left, right in zip(result["segments"], result["segments"][1:]):
                self.assertLess(left["end_ms"], right["start_ms"])
            self.assertLessEqual(result["total_duration_ms"], budget)

    def test_full_twelve_candidates_case(self):
        channels = ["m", "s"] * 6
        commands = [
            cmd(f"c{i:02d}", channel=channels[i], offset=i * 3, duration=5)
            for i in range(12)
        ]
        cands = [candidate(f"c{i:02d}", i % 4) for i in range(12)]
        result = plan_replays(commands, cands, 2, 2, 100)
        expected = brute_force(commands, cands, 2, 2, 100)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
