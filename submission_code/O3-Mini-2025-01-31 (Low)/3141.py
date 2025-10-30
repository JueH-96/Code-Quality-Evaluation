from typing import List
import bisect

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        cycle_sum = sum(nums)
        
        # Precompute prefix sums for two copies of nums.
        # P[0] = 0, P[i] = sum of first i elements.
        double_n = nums + nums
        m = len(double_n)
        P = [0] * (m + 1)
        for i in range(m):
            P[i+1] = P[i] + double_n[i]
        # P is strictly increasing
        
        # Build a dictionary mapping prefix sum -> index
        # Because P is strictly increasing, each value appears once.
        pos = {val: i for i, val in enumerate(P)}
        
        # For a given required partial sum (rem),
        # we wish to find (over starting indices i in [0,n]) the minimal window length L
        # such that there is j with P[j] - P[i] == rem.
        def min_partial_length(rem: int) -> int:
            best = float('inf')
            # We only need to consider starting indices in the first copy (i in [0, n])
            for i in range(n+1):
                # We need j such that P[j] == P[i] + rem.
                want = P[i] + rem
                # Binary search in P since it is sorted.
                # Alternatively, use the dictionary we built.
                j = pos.get(want, None)
                if j is not None and j - i <= m - i:  # automatically true; we restrict
                    best = min(best, j - i)
            return best if best != float('inf') else None
        
        ans = float('inf')
        # Try the case rem = 0 separately.
        if target % cycle_sum == 0:
            ans = min(ans, (target // cycle_sum) * n)
        
        # The candidate k (full cycles) can be from 0 to target//cycle_sum.
        kmax = target // cycle_sum
        for k in range(kmax + 1):
            rem = target - k * cycle_sum
            if rem < 0:
                continue
            # Find minimal contiguous window (from at most 2 copies) that sums to rem.
            L = min_partial_length(rem)
            if L is not None:
                candidate = k * n + L
                ans = min(ans, candidate)
        return ans if ans != float('inf') else -1

# You can include some test cases to verify the solution.
if __name__ == "__main__":
    sol = Solution()
    # Example 1: nums=[1,2,3], target=5 -> answer 2 (subarray [2,3] or [3,1,?] but 2 is best)
    print(sol.minSizeSubarray([1,2,3], 5))   # Expected output: 2
    # Example 2: nums=[1,1,1,2,3], target=4 -> answer 2 ([1,3] or [2,?])
    print(sol.minSizeSubarray([1,1,1,2,3], 4))   # Expected output: 2
    # Example 3: nums=[2,4,6,8], target=3 -> answer -1 (no subarray)
    print(sol.minSizeSubarray([2,4,6,8], 3))   # Expected output: -1