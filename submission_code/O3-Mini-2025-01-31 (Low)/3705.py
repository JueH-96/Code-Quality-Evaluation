from typing import List

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        subarray_count = {}
        
        # For each subarray of length k, add each unique integer once to its count.
        for i in range(n - k + 1):
            window = nums[i:i+k]
            seen = set(window)
            for num in seen:
                subarray_count[num] = subarray_count.get(num, 0) + 1
        
        # Collect numbers that appear in exactly one subarray.
        almost_missing = [num for num, count in subarray_count.items() if count == 1]
        
        # Return the largest number if there is any, else return -1.
        return max(almost_missing) if almost_missing else -1