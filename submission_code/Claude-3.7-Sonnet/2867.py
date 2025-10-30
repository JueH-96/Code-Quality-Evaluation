class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find the first occurrence of 1
        first_one = -1
        for i, num in enumerate(nums):
            if num == 1:
                first_one = i
                break
        
        # If there are no 1s, we can't form any good subarrays
        if first_one == -1:
            return 0
        
        result = 1
        prev_one = first_one
        
        # For each subsequent 1, calculate the number of ways to split
        # between the previous 1 and the current 1
        for i in range(first_one + 1, len(nums)):
            if nums[i] == 1:
                # We can place a split at any position between prev_one and i
                # That gives us (i - prev_one) possible positions
                result = (result * (i - prev_one)) % MOD
                prev_one = i
        
        return result