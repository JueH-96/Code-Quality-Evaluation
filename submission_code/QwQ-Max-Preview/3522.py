from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            sub = nums[i:i + k]
            # Check if the subarray is sorted in strictly increasing order
            is_sorted = True
            for j in range(len(sub) - 1):
                if sub[j] >= sub[j + 1]:
                    is_sorted = False
                    break
            if not is_sorted:
                result.append(-1)
                continue
            # Check if elements are consecutive
            first = sub[0]
            last = sub[-1]
            if last - first == k - 1:
                result.append(last)
            else:
                result.append(-1)
        return result