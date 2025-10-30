from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        required_indices = set(range(1, n + 1))
        available_indices = set(changeIndices)
        if not required_indices.issubset(available_indices):
            return -1
        
        def is_possible(seconds):
            if seconds < n:
                return False
            last_occurrence = {}
            indices_in_range = set()
            for i in range(seconds):
                index = changeIndices[i]
                last_occurrence[index] = i + 1
                indices_in_range.add(index)
            for i in range(1, n + 1):
                if i not in indices_in_range:
                    return False
            if sum(nums) <= seconds - n:
                return True
            return False

        low = n
        high = m
        earliest_second = -1

        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                earliest_second = mid
                high = mid - 1
            else:
                low = mid + 1

        return earliest_second