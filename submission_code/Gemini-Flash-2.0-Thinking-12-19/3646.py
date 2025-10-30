import collections

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        dp_sum = collections.defaultdict(int)
        dp_count = collections.defaultdict(int)
        mod = 10**9 + 7
        for num in nums:
            current_sum = num
            current_count = 1
            if num - 1 in dp_sum:
                current_sum = (current_sum + dp_sum[num - 1] + dp_count[num - 1] * num) % mod
                current_count = (current_count + dp_count[num - 1]) % mod
            dp_sum[num] = (dp_sum[num] + current_sum) % mod
            dp_count[num] = (dp_count[num] + current_count) % mod
        
        final_sum = 0
        for val in dp_sum:
            final_sum = (final_sum + dp_sum[val]) % mod
        return final_sum