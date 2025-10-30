from typing import List

MOD = 10 ** 9 + 7


class Solution:
    # ------------------------------------------------------------------
    # how many ways to pick k indices whose pairwise distance is >= gap
    # ------------------------------------------------------------------
    def _count_with_gap(self, arr: List[int], k: int, gap: int) -> int:
        n = len(arr)
        # dp[p][c] = number of ways to choose c indices from first p elements
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        left = 0                       # left points to first element that is < gap away from arr[i]
        for i in range(n):
            while left < i and arr[i] - arr[left] >= gap:
                left += 1

            for c in range(k + 1):
                if dp[i][c] == 0:
                    continue
                # 1) skip element i
                dp[i + 1][c] = (dp[i + 1][c] + dp[i][c]) % MOD
                # 2) take element i  (choose remaining c from first `left` elements)
                if c + 1 <= k:
                    dp[i + 1][c + 1] = (dp[i + 1][c + 1] + dp[left][c]) % MOD

        return dp[n][k]

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)

        # collect every possible distance that can become the minimum
        diffs = {0}
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(nums[j] - nums[i])

        diffs = sorted(diffs)
        diffs.append(diffs[-1] + 1)         # sentinel that makes the count zero

        # pre–compute F(d) –– number of k-subsets whose minimum distance ≥ d
        F = []
        for d in diffs:
            if d == diffs[-1]:              # sentinel ⇒ no valid subset
                F.append(0)
            else:
                F.append(self._count_with_gap(nums, k, d))

        # Sum up contributions
        ans = 0
        for i in range(len(diffs) - 1):
            exact_cnt = (F[i] - F[i + 1]) % MOD        # subsets whose min distance equals diffs[i]
            ans = (ans + (diffs[i] % MOD) * exact_cnt) % MOD

        return ans