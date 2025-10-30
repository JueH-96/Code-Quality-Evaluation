from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        def is_strictly_increasing(arr: List[int]) -> bool:
            # An empty or singleton array is strictly increasing by definition
            for i in range(1, len(arr)):
                if arr[i] <= arr[i-1]:
                    return False
            return True

        count = 0
        # Enumerate all non-empty subarrays [i..j]
        for i in range(n):
            for j in range(i, n):
                # Build array after removing nums[i..j]
                new_arr = nums[:i] + nums[j+1:]
                if is_strictly_increasing(new_arr):
                    count += 1
        return count