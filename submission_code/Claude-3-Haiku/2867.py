class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        # Count the number of 1's in the array
        ones = nums.count(1)
        
        # If there are less than 2 ones, there are no good subarrays
        if ones < 2:
            return 0
        
        # Initialize the count of good subarrays to 1 (the entire array)
        count = 1
        
        # Iterate through the array and count the number of good subarrays
        prev = -1
        for i in range(n):
            if nums[i] == 1:
                count = (count * (i - prev)) % mod
                prev = i
        
        return count