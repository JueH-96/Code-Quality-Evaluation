from typing import List
import math

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] is the maximum number of jumps to reach index i.
        dp = [-math.inf] * n
        dp[0] = 0  # starting at index 0 with 0 jumps
        
        # Try all jumps from i to j (where i < j)
        for i in range(n):
            # Only consider i if it is reachable
            if dp[i] == -math.inf:
                continue
            for j in range(i+1, n):
                # Check if we can jump from index i to j
                if -target <= nums[j] - nums[i] <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
                    
        return dp[n-1] if dp[n-1] != -math.inf else -1


# Sample test cases
if __name__ == '__main__':
    sol = Solution()
    print(sol.maximumJumps([1, 3, 6, 4, 1, 2], 2))  # Output: 3
    print(sol.maximumJumps([1, 3, 6, 4, 1, 2], 3))  # Output: 5
    print(sol.maximumJumps([1, 3, 6, 4, 1, 2], 0))  # Output: -1