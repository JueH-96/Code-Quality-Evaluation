from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        result = [0] * n
        i = 0
        
        while i < n:
            group = [sorted_nums[i][1]]
            j = i + 1
            while j < n and sorted_nums[j][0] - sorted_nums[i][0] <= limit:
                group.append(sorted_nums[j][1])
                j += 1
            for idx in sorted(group):
                result[idx] = sorted_nums[i][0]
                i += 1
        return result