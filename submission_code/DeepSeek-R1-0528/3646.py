class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        max_val = max(nums)
        dp_count = [0] * (max_val + 2)
        dp_sum = [0] * (max_val + 2)
        
        for a in nums:
            count_here = 1
            sum_here = a
            
            if a - 1 >= 0:
                count_here = (count_here + dp_count[a-1]) % mod
                sum_here = (sum_here + dp_sum[a-1] + dp_count[a-1] * a) % mod
            
            if a + 1 <= max_val + 1:
                count_here = (count_here + dp_count[a+1]) % mod
                sum_here = (sum_here + dp_sum[a+1] + dp_count[a+1] * a) % mod
            
            dp_count[a] = (dp_count[a] + count_here) % mod
            dp_sum[a] = (dp_sum[a] + sum_here) % mod
        
        total = 0
        for s in dp_sum:
            total = (total + s) % mod
        return total