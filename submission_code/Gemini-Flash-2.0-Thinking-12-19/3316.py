from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        possible_diffs = set()
        possible_diffs.add(0)
        for i in range(n):
            for j in range(i + 1, n):
                possible_diffs.add(nums[j] - nums[i])
        sorted_diffs = sorted(list(possible_diffs))
        unique_diffs = []
        if sorted_diffs:
            unique_diffs.append(sorted_diffs[0])
            for i in range(1, len(sorted_diffs)):
                if sorted_diffs[i] > sorted_diffs[i-1]:
                    unique_diffs.append(sorted_diffs[i])
        
        diff_values = unique_diffs
        if not diff_values:
            diff_values = [0]
        
        def count_subsequences_at_least_power(d):
            dp = [[0] * (k + 1) for _ in range(n)]
            for i in range(n):
                dp[i][1] = 1
            for j in range(2, k + 1):
                for i in range(j - 1, n):
                    for l in range(j - 2, i):
                        if nums[i] - nums[l] >= d:
                            dp[i][j] = (dp[i][j] + dp[l][j - 1]) % (10**9 + 7)
            count = 0
            for i in range(k - 1, n):
                count = (count + dp[i][k]) % (10**9 + 7)
            return count
            
        total_power_sum = 0
        prev_diff = 0
        for diff in diff_values:
            if diff < prev_diff:
                continue
            count = count_subsequences_at_least_power(diff)
            delta_diff = diff - prev_diff
            total_power_sum = (total_power_sum + (delta_diff * count) % (10**9 + 7)) % (10**9 + 7)
            prev_diff = diff
            
        return total_power_sum