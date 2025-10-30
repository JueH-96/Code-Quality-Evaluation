class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        def count_ways(num):
            return (num + 1) * (num + 2) // 2
        
        total = 1  # Initialize with 1 for the case where both arrays are [0, 0, ..., 0]
        
        for i in range(n):
            min_val = 0 if i == 0 else nums[i] - nums[i-1]
            max_val = nums[i]
            
            ways = count_ways(max_val - min_val)
            total = (total * ways) % MOD
        
        return total