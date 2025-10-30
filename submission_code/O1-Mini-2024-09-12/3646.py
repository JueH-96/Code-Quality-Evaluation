class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        
        max_num = max(nums)
        size = max_num + 2  # To handle num+1 indexing
        counts = [0] * size
        sums = [0] * size
        
        total_sum = 0
        
        for num in nums:
            # Current counts and sums before updating
            count_prev = counts[num -1] if num -1 >=0 else 0
            count_next = counts[num +1] if num +1 < size else 0
            sum_prev = sums[num -1] if num -1 >=0 else 0
            sum_next = sums[num +1] if num +1 < size else 0
            
            # Update counts
            new_count = (1 + count_prev + count_next) % MOD
            counts[num] = (counts[num] + new_count) % MOD
            
            # Update sums
            new_sum = (num + sum_prev + num * count_prev + sum_next + num * count_next) % MOD
            sums[num] = (sums[num] + new_sum) % MOD
            
        # The total sum is the sum of all sums
        total_sum = sum(sums) % MOD
        return total_sum