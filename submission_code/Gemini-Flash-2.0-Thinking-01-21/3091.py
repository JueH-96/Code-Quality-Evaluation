from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        counts = Counter(nums)
        distinct_nums_counts = list(counts.items())
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        mod = 10**9 + 7
        for num, count in distinct_nums_counts:
            next_dp = [0] * (max_sum + 1)
            for current_sum in range(max_sum + 1):
                for n_copies in range(count + 1):
                    prev_sum = current_sum - n_copies * num
                    if prev_sum >= 0:
                        next_dp[current_sum] = (next_dp[current_sum] + dp[prev_sum]) % mod
            dp = next_dp
        
        result = 0
        for s in range(l, r + 1):
            result = (result + dp[s]) % mod
            
        return result