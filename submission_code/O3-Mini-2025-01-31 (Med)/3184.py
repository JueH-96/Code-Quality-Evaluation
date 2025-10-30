import bisect
from typing import List

# We'll use a Fenwick Tree (Binary Indexed Tree) to support range maximum queries.
# This is needed to efficiently find, for a given u = nums[i] - i, the maximum dp value
# that we have seen for all indices j with u[j] <= u (because the balanced condition 
# simplifies to u[j] <= u[i] for a valid transition from j to i).

class Fenw:
    def __init__(self, n: int):
        self.n = n
        self.data = [-10**18] * (n + 1)  # Use a very small number as -infinity
    
    def update(self, i: int, val: int) -> None:
        # i is 0-indexed; convert it to 1-indexed for Fenw tree operations.
        i += 1
        while i <= self.n:
            self.data[i] = max(self.data[i], val)
            i += i & -i
    
    def query(self, i: int) -> int:
        res = -10**18
        i += 1
        while i > 0:
            res = max(res, self.data[i])
            i -= i & -i
        return res
        
class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # For each index compute u = nums[i] - i.
        u_vals = [nums[i] - i for i in range(n)]
        
        # Coordinate compress u values.
        sorted_unique = sorted(set(u_vals))
        
        # Create a Fenw tree with size equal to the number of unique u values.
        fenw = Fenw(len(sorted_unique))
        
        best_answer = -10**18
        
        # Process indices in increasing order (which is the natural order for a subsequence).
        for i in range(n):
            u = nums[i] - i
            # Determine the compressed index (0-indexed) for u.
            pos = bisect.bisect_left(sorted_unique, u)
            # Query the maximum dp value for any previous index j with u[j] <= u.
            bestPrev = fenw.query(pos)
            # A balanced subsequence ending at i can either start fresh (only nums[i])
            # or extend a previous valid sequence.  If bestPrev is negative (or -infty),
            # max(0, bestPrev) ensures that starting fresh is chosen.
            cur_dp = nums[i] + max(0, bestPrev)
            
            best_answer = max(best_answer, cur_dp)
            # Update the Fenw tree for the current u value.
            fenw.update(pos, cur_dp)
            
        return best_answer

# Example usage and test cases:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxBalancedSubsequenceSum([3,3,5,6]))   # Expected output: 14
    print(sol.maxBalancedSubsequenceSum([5,-1,-3,8]))   # Expected output: 13
    print(sol.maxBalancedSubsequenceSum([-2,-1]))       # Expected output: -1