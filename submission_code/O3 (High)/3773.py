from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        """
        Repeatedly replace the left-most adjacent pair that has the smallest
        sum until the array becomes non-decreasing.  Return the number of
        replacements performed.
        """
        # Helper: check if current list is non-decreasing
        def is_sorted(arr: List[int]) -> bool:
            return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

        # If already non-decreasing or has < 2 elements, no operation needed
        if len(nums) < 2 or is_sorted(nums):
            return 0

        nums = nums[:]                    # work on a copy
        operations = 0

        while not is_sorted(nums):
            # 1. locate left-most adjacent pair with the minimum sum
            min_sum = float('inf')
            min_idx = 0
            for i in range(len(nums) - 1):
                s = nums[i] + nums[i + 1]
                if s < min_sum:
                    min_sum = s
                    min_idx = i
            # 2. replace that pair with their sum
            nums[min_idx] = min_sum
            nums.pop(min_idx + 1)
            operations += 1

        return operations