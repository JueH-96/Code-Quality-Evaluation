from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # prefix sums of nums and cost (inclusive)
        pre_num = [0] * n
        pre_cost = [0] * n
        s = 0
        for i, v in enumerate(nums):
            s += v
            pre_num[i] = s
        s = 0
        for i, v in enumerate(cost):
            s += v
            pre_cost[i] = s

        INF = 10 ** 30

        # dp_prev[r]  -> minimal cost to cover indices 0..r with (seg-1) segments
        dp_prev = [(pre_num[r] + k) * pre_cost[r] for r in range(n)]
        answer = dp_prev[-1]

        # helper functions for Convex Hull
        def add_line(hull: List[tuple], m: int, b: int) -> None:
            """
            Insert a new line (m, b) into the hull.
            Slopes are added in strictly decreasing order.
            """
            # while last line is made obsolete by the new one, remove it
            while len(hull) >= 2:
                m1, b1 = hull[-2]
                m2, b2 = hull[-1]
                # if intersection(L1,L2) >= intersection(L2,new)  -> remove L2
                # (b2 - b1)/(m1 - m2) >= (b  - b2)/(m2 - m)
                if (b2 - b1) * (m2 - m) >= (b - b2) * (m1 - m2):
                    hull.pop()
                else:
                    break
            hull.append((m, b))

        def query(hull: List[tuple], x: int, ptr: int) -> (int, int):
            """
            Return minimal value at x and updated pointer.
            Queries come with non-decreasing x, slopes in hull
            are strictly decreasing ⇒ we can advance pointer.
            """
            while ptr + 1 < len(hull) and hull[ptr][0] * x + hull[ptr][1] >= hull[ptr + 1][0] * x + hull[ptr + 1][1]:
                ptr += 1
            return hull[ptr][0] * x + hull[ptr][1], ptr

        # iterate over number of segments (seg)
        for seg in range(2, n + 1):
            dp_curr = [INF] * n
            hull: List[tuple] = []
            ptr = 0  # pointer for monotone queries

            # iterate over right end of current segment
            for r in range(seg - 1, n):
                j = r - 1  # previous segment ends at j
                if dp_prev[j] < INF:
                    # line: y = m*x + b   with  m = -pre_cost[j]
                    m = -pre_cost[j]
                    b = dp_prev[j] - k * seg * pre_cost[j]
                    add_line(hull, m, b)

                x = pre_num[r]
                best, ptr = query(hull, x, ptr)
                dp_curr[r] = pre_num[r] * pre_cost[r] + k * seg * pre_cost[r] + best

            answer = min(answer, dp_curr[-1])
            dp_prev = dp_curr

        return answer