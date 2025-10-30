from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        marked = [False] * n
        earliest = [float('inf')] * n
        
        def can_mark_all_in(seconds):
            nonlocal earliest
            count = 0
            for i in range(seconds - 1, -1, -1):
                index = changeIndices[i] - 1
                if not marked[index]:
                    if nums[index] <= seconds - i - 1:
                        count += 1
                        marked[index] = True
                        if count == n:
                            return True
                    else:
                        break
            for i in range(seconds - 1, -1, -1):
                index = changeIndices[i] - 1
                if marked[index]:
                    nums[index] -= seconds - i - 1
                    if nums[index] == 0:
                        count -= 1
                        marked[index] = False
            return count == n
        
        for i in range(m):
            index = changeIndices[i] - 1
            if nums[index] > 0:
                earliest[index] = min(earliest[index], i + 1)
        
        left, right = 1, m
        while left < right:
            mid = (left + right) // 2
            if can_mark_all_in(mid):
                right = mid
            else:
                left = mid + 1
        
        return left if can_mark_all_in(left) else -1