from typing import List
import bisect

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        def can_mark_all(seconds):
            # Track the last occurrence of each index in changeIndices within the given seconds
            last_occurrence = [-1] * n
            for s in range(seconds):
                last_occurrence[changeIndices[s] - 1] = s
            
            # If any index never appears in changeIndices within the given seconds, return False
            if -1 in last_occurrence:
                return False
            
            # Track the number of operations we can use to decrement nums
            operations = 0
            marked = [False] * n
            
            for s in range(seconds - 1, -1, -1):
                index = changeIndices[s] - 1
                if not marked[index]:
                    # Use operations to decrement nums[index] to 0
                    if operations >= nums[index]:
                        operations -= nums[index]
                        marked[index] = True
                    else:
                        # If we don't have enough operations, use the current second to decrement
                        operations += 1
                else:
                    # If the index is already marked, we can use this second to decrement any index
                    operations += 1
            
            # Check if all indices are marked
            return all(marked)
        
        # Binary search for the earliest second
        left, right = 0, m + 1
        while left < right:
            mid = (left + right) // 2
            if can_mark_all(mid):
                right = mid
            else:
                left = mid + 1
        
        return left if left <= m else -1