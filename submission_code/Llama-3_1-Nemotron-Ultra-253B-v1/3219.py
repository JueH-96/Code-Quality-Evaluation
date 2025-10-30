from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        sorted_nums = sorted((val, idx) for idx, val in enumerate(nums))
        n = len(nums)
        result = [0] * n
        i = 0
        while i < n:
            group = []
            group.append(sorted_nums[i])
            j = i + 1
            while j < n:
                if sorted_nums[j][0] - sorted_nums[j-1][0] <= limit:
                    group.append(sorted_nums[j])
                    j += 1
                else:
                    break
            indices = [idx for (val, idx) in group]
            indices.sort()
            vals = [val for (val, idx) in group]
            for k in range(len(indices)):
                result[indices[k]] = vals[k]
            i = j
        return result