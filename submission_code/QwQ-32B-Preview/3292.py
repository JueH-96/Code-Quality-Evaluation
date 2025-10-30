from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        changeIndices = [idx - 1 for idx in changeIndices]  # Adjust to 0-indexing

        # Helper function to check if it's possible to mark all indices by time T
        def is_possible(T):
            # Find the last available second to mark each index up to T
            last_available = [0] * n
            for s in range(T - 1, -1, -1):
                idx = changeIndices[s]
                if last_available[idx] == 0:
                    last_available[idx] = s + 1  # Convert back to 1-indexing

            # Check if all indices are present in changeIndices up to T
            for i in range(n):
                if last_available[i] == 0:
                    return False

            # Collect indices with their last available seconds
            indices_with_deadlines = [(last_available[i], nums[i]) for i in range(n)]
            # Sort by last available second in ascending order
            indices_with_deadlines.sort()

            # Check if all decrements can be scheduled within their deadlines
            total_decrements_needed = 0
            for deadline, decrements in indices_with_deadlines:
                total_decrements_needed += decrements
                if total_decrements_needed > deadline:
                    return False
            return True

        # Binary search over possible T from 1 to m
        low = 1
        high = m
        ans = -1
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans