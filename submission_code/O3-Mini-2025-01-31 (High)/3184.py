from typing import List
import bisect

class FenwTree:
    def __init__(self, size: int):
        self.size = size
        # Initialize all tree positions with 0.
        # 0 here serves as our identity so that if no candidate has been added,
        # the best value we get is 0, so that dp[i] = nums[i] + max(0, candidate)
        self.data = [0] * (size + 1)
        
    def update(self, i: int, value: int):
        # i is 1-indexed in the FenwTree.
        while i <= self.size:
            if value > self.data[i]:
                self.data[i] = value
            i += i & -i
            
    def query(self, i: int) -> int:
        # Returns the maximum value in the prefix [1, i]
        res = 0
        while i:
            if self.data[i] > res:
                res = self.data[i]
            i -= i & -i
        return res

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Define key for each index as: key = nums[i] - i.
        # The balanced subsequence condition for two consecutive indices i and j is:
        #   nums[j] - nums[i] >= j - i   <=>   (nums[j] - j) >= (nums[i] - i)
        # Thus, when appending to a subsequence ending at some index i, we require that
        # key[i] <= key[j].
        keys = [nums[i] - i for i in range(n)]
        
        # Coordinate-compress the keys as they can be large integers.
        sorted_unique = sorted(set(keys))
        comp = {val: idx + 1 for idx, val in enumerate(sorted_unique)}  # using 1-indexed positions
        
        # Initialize a FenwTree to support range maximum queries.
        ft = FenwTree(len(sorted_unique))
        
        # dp[j] = maximum balanced subsequence sum ending at index j.
        # Recurrence: dp[j] = nums[j] + max(0, max{ dp[i] for all i < j with keys[i] <= keys[j] })
        ans = -10**18
        for i in range(n):
            # For current element, get its compressed key.
            c = comp[keys[i]]
            best = ft.query(c)
            # We only add the best candidate if it is positive; otherwise a subsequence of length 1 is best.
            dp_val = nums[i] + (best if best > 0 else 0)
            ft.update(c, dp_val)
            if dp_val > ans:
                ans = dp_val
        
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxBalancedSubsequenceSum([3,3,5,6]))   # Expected output: 14
    print(sol.maxBalancedSubsequenceSum([5,-1,-3,8]))   # Expected output: 13
    print(sol.maxBalancedSubsequenceSum([-2,-1]))       # Expected output: -1