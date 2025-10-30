from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        if k > n:
            return 0
        differences = set()
        for i in range(n):
            for j in range(i + 1, n):
                differences.add(sorted_nums[j] - sorted_nums[i])
        distinct_differences = sorted(list(differences))
        if not distinct_differences:
            return 0
        D = [0] + distinct_differences
        m = len(distinct_differences)
        total_sum_of_powers = 0
        mod = 10**9 + 7
        
        for r in range(1, m + 1):
            current_diff = distinct_differences[r-1]
            dp = [[0] * n for _ in range(k + 1)]
            for v in range(n):
                dp[1][v] = 1
            for length in range(2, k + 1):
                for v in range(length - 1, n):
                    for u in range(length - 2, v):
                        if sorted_nums[v] - sorted_nums[u] >= current_diff:
                            dp[length][v] = (dp[length][v] + dp[length - 1][u]) % mod
            count_subsequences = 0
            for v in range(k - 1, n):
                count_subsequences = (count_subsequences + dp[k][v]) % mod
            delta_d = distinct_differences[r-1] - (distinct_differences[r-2] if r > 1 else 0)
            contribution = (delta_d * count_subsequences) % mod
            total_sum_of_powers = (total_sum_of_powers + contribution) % mod
            
        return total_sum_of_powers