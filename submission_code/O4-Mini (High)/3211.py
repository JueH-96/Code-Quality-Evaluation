from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        # Build prefix sums P where P[i] = sum of nums[0..i-1]
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + nums[i]
        
        # Check if it's possible to split nums into t non-decreasing-sum segments
        def ok(t: int) -> bool:
            prev = 0    # sum of the previous segment
            pos = 0     # current start index for the next segment in nums
            j = 0       # scanning pointer for finding segment end
            
            # We will choose the first t-1 segments greedily (smallest end index
            # so that each segment sum >= prev), leaving the remainder for the last.
            for k in range(1, t):
                # Ensure j starts at least at pos
                j = max(j, pos)
                # We must leave at least (t-k) elements for the remaining segments,
                # each of size >= 1. So the largest j we can pick now is:
                j_end = n - t + k - 1
                if j > j_end:
                    return False
                # Advance j until the segment [pos..j] has sum >= prev
                # i.e. P[j+1] - P[pos] >= prev
                while j <= j_end and (P[j+1] - P[pos]) < prev:
                    j += 1
                if j > j_end:
                    return False
                # Accept this segment
                prev = P[j+1] - P[pos]
                pos = j + 1
            
            # Last segment covers [pos..n-1], must also have sum >= prev
            if pos >= n:
                return False
            if (P[n] - P[pos]) < prev:
                return False
            return True
        
        # Binary search for the largest t in [1..n] for which ok(t) is True
        low, high = 1, n
        ans = 1
        while low <= high:
            mid = (low + high) // 2
            if ok(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans