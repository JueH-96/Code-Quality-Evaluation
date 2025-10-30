from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            # Check if the subarray is sorted in ascending order
            is_sorted = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j + 1]:
                    is_sorted = False
                    break
            if is_sorted:
                # Check if the elements are consecutive
                if nums[i + k - 1] - nums[i] == k - 1:
                    result.append(nums[i + k - 1])
                else:
                    result.append(-1)
            else:
                result.append(-1)
        return result