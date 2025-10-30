from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = (n + 2) // 3  # Calculate ceil(n/3)
        for k in range(max_k + 1):  # Iterate k from 0 to max_k inclusive
            s = min(3 * k, n)  # Start index of the remaining subarray
            subarray = nums[s:]  # Slice the subarray starting from index s
            if len(set(subarray)) == len(subarray):  # Check if all elements are distinct
                return k  # Return the number of operations