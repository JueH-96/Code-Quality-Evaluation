from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        total_sum = sum(nums)
        
        # Check if it's possible to mark all indices
        if len(set(changeIndices)) < n:
            return -1
        
        # Binary search for the earliest second
        left, right = 1, m
        while left < right:
            mid = (left + right) // 2
            marked = [False] * n
            available = mid
            
            # Try to mark indices up to the mid second
            for i in range(mid - 1, -1, -1):
                idx = changeIndices[i] - 1
                if not marked[idx]:
                    if available >= nums[idx]:
                        available -= nums[idx]
                        marked[idx] = True
                    else:
                        available -= 1
                else:
                    available -= 1
            
            if all(marked):
                right = mid
            else:
                left = mid + 1
        
        return left if all(marked) else -1