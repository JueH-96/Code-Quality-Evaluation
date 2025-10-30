class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        sum_end = {}
        count_end = {}
        
        total_sum = 0
        
        for num in nums:
            extend_sum = sum_end.get(num - 1, 0) + sum_end.get(num + 1, 0)
            extend_count = count_end.get(num - 1, 0) + count_end.get(num + 1, 0)
            
            new_sum = (num * (1 + extend_count) + extend_sum) % MOD
            sum_end[num] = (sum_end.get(num, 0) + new_sum) % MOD
            count_end[num] = (count_end.get(num, 0) + 1 + extend_count) % MOD
            
            total_sum = (total_sum + new_sum) % MOD
        
        return total_sum