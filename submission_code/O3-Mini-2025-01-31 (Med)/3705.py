from typing import List

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Dictionary to count the number of subarrays
        # in which each integer appears
        count = {}
        
        n = len(nums)
        # Loop over each subarray of size k
        for i in range(n - k + 1):
            # Use a set to ensure we count a number only once per subarray
            subarray_unique = set(nums[i:i+k])
            for x in subarray_unique:
                count[x] = count.get(x, 0) + 1
                
        # Filter for those integers that appear exactly in one subarray
        candidates = [x for x in count if count[x] == 1]
        
        # If no such integer exists, return -1, otherwise return the largest
        return max(candidates) if candidates else -1