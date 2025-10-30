from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        # fixed_max: maximum absolute difference between adjacent fixed (non-missing) numbers.
        fixed_max = 0
        for i in range(1, n):
            if nums[i] != -1 and nums[i-1] != -1:
                fixed_max = max(fixed_max, abs(nums[i] - nums[i-1]))
        
        # Collect all fixed numbers that are adjacent to at least one missing position.
        boundaries = []
        for i in range(n):
            if nums[i] == -1:
                continue
            if (i > 0 and nums[i-1] == -1) or (i < n - 1 and nums[i+1] == -1):
                boundaries.append(nums[i])
                
        # If there are no boundaries, then no adjacent difference from missing boundaries exists.
        # We can simply use any pair (k, k) so that all missing are replaced with k.
        if not boundaries:
            return fixed_max
        
        # Sort boundaries so that we can cover them with two intervals.
        boundaries.sort()
        
        # Helper function: given candidate M, check if it is possible to choose two numbers
        # (which yield intervals of coverage [a - M, a+M] and [b - M, b+M]) so that every value in boundaries
        # is within at least one of these two intervals.
        # Because the intervals are continuous in value space, and boundaries is sorted,
        # the union of two intervals is at most two contiguous intervals.
        # So, we can try to partition boundaries into 2 contiguous segments.
        def canCover(M: int) -> bool:
            m = len(boundaries)
            # Try every partition point k in [0, m] where:
            #   - if k == 0, then the second interval must cover all boundaries.
            #   - if k == m, then the first interval covers all boundaries.
            #   - otherwise, first covers boundaries[0..k-1] and second covers boundaries[k..m-1].
            # For an interval, if the difference between its min and max is <= 2*M, then there exists a center
            # number that covers all values in that interval.
            # Since boundaries is sorted, a contiguous segment is coverable if boundaries[last] - boundaries[first] <= 2*M.
            # Check for partition k = 0 (first interval empty, second covers all)
            if boundaries[-1] - boundaries[0] <= 2 * M:
                return True
            # Otherwise, try splitting the boundaries list.
            for k in range(1, m):
                if boundaries[k-1] - boundaries[0] <= 2 * M and boundaries[-1] - boundaries[k] <= 2 * M:
                    return True
            return False
        
        # Binary search for the minimal M required
        low = 0
        high = boundaries[-1] - boundaries[0]  # worst-case: one interval covering full span
        candidate = high
        
        while low <= high:
            mid = (low + high) // 2
            if canCover(mid):
                candidate = mid
                high = mid - 1
            else:
                low = mid + 1
        
        # The answer must be at least fixed_max because of fixed adjacent differences.
        return max(fixed_max, candidate)