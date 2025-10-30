from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_elem = max(nums)
        count = 0
        n = len(nums)
        i = 0

        while i < n:
            if nums[i] == max_elem:
                j = i
                max_count = 0
                while j < n and nums[j] <= max_elem:
                    if nums[j] == max_elem:
                        max_count += 1
                    if max_count >= k:
                        count += (j - i) - (max_count - k) + 1
                    j += 1
                i = j - 1
            i += 1

        return count