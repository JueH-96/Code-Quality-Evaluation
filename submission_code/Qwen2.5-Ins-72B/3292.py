from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        def can_mark_all_by_second(s: int) -> bool:
            marked = [False] * n
            operations = 0
            for i in range(s - 1, -1, -1):
                idx = changeIndices[i] - 1
                if not marked[idx]:
                    if nums[idx] <= operations:
                        marked[idx] = True
                        operations -= nums[idx]
                    else:
                        return False
                operations += 1
            return all(marked)
        
        for s in range(1, m + 1):
            if can_mark_all_by_second(s):
                return s
        return -1