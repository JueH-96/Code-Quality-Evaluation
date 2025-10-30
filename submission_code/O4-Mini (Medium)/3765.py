from typing import List

class ConvexHullTrick:
    # Maintains lower hull for lines added in order of non-increasing slopes,
    # supports queries at non-decreasing x.
    def __init__(self):
        self.ms = []
        self.bs = []
        self.ptr = 0

    def add_line(self, m: int, b: int) -> None:
        # remove last while the new line makes it obsolete
        # we want to check intersection of (m1,b1) & (m2,b2) >= intersection of (m2,b2) & (m3,b3)
        # where m1>=m2>=m3
        while len(self.ms) >= 2:
            m1, b1 = self.ms[-2], self.bs[-2]
            m2, b2 = self.ms[-1], self.bs[-1]
            m3, b3 = m, self.bs + [b]  # temporarily build for clarity
            # Actually b3 is b; we don't need bs+[b]
            b3 = b
            # check (b2 - b1)/(m1 - m2) >= (b3 - b2)/(m2 - m3)
            # cross-multiplied to avoid floating point:
            # (b2 - b1)*(m2 - m3) >= (b3 - b2)*(m1 - m2)
            if (b2 - b1) * (m2 - m3) >= (b3 - b2) * (m1 - m2):
                # remove the middle line
                self.ms.pop()
                self.bs.pop()
            else:
                break
        self.ms.append(m)
        self.bs.append(b)
        # adjust pointer if out of bounds
        if self.ptr >= len(self.ms):
            self.ptr = len(self.ms) - 1

    def query(self, x: int) -> int:
        # return min value at x
        # advance ptr while next line gives smaller value
        while self.ptr + 1 < len(self.ms):
            v1 = self.ms[self.ptr] * x + self.bs[self.ptr]
            v2 = self.ms[self.ptr + 1] * x + self.bs[self.ptr + 1]
            if v2 <= v1:
                self.ptr += 1
            else:
                break
        return self.ms[self.ptr] * x + self.bs[self.ptr]


class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # prefix sums
        P = [0] * n
        C = [0] * n
        P[0] = nums[0]
        C[0] = cost[0]
        for i in range(1, n):
            P[i] = P[i-1] + nums[i]
            C[i] = C[i-1] + cost[i]

        INF = 10**36
        # dp_prev[r] will hold dp[r][t-1], dp_cur[r] dp[r][t]
        dp_prev = [INF] * n
        # t = 1 base case: only one segment from 0..r
        for r in range(n):
            dp_prev[r] = C[r] * (P[r] + k)

        ans = dp_prev[n-1]
        # for segments t = 2..n
        for t in range(2, n+1):
            cht = ConvexHullTrick()
            dp_cur = [INF] * n
            # we add lines from dp_prev after computing dp_cur[r]
            for r in range(n):
                # query only if there is at least one line
                if cht.ms:
                    best = cht.query(P[r])
                    dp_cur[r] = C[r] * (P[r] + k * t) + best
                # now add line from dp_prev[r] for future r'
                val = dp_prev[r]
                if val < INF:
                    # slope = -C[r], intercept = dp_prev[r] - C[r]*k*t
                    m = -C[r]
                    b = val - C[r] * (k * t)
                    cht.add_line(m, b)
            dp_prev = dp_cur
            if dp_prev[n-1] < ans:
                ans = dp_prev[n-1]
        return ans