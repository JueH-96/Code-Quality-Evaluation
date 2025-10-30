from typing import List
import math

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # prefix sum array: ps[i] = sum(nums[0:i])
        ps = [0] * (n + 1)
        for i in range(1, n + 1):
            ps[i] = ps[i - 1] + nums[i - 1]
        
        # Initialize dp table: dp[j][i] = maximum sum using j segments in first i elements
        # We'll use dp[j][i] for 0 <= j <= k and 0 <= i <= n.
        # Base: dp[0][i] = 0 for all i
        dp = [[-math.inf] * (n + 1) for _ in range(k + 1)]
        for i in range(n + 1):
            dp[0][i] = 0
        
        # For each number of segments from 1 to k
        for seg in range(1, k + 1):
            # For dp[seg][i] to be valid, we need at least seg*m elements.
            # We'll also maintain a helper array M, where:
            #   M[i] = max_{x in [0, i - m]} (dp[seg-1][x] - ps[x]), for i >= m.
            # We can build this cumulatively as we increase i.
            M = [-math.inf] * (n + 1)
            # We'll update M starting from i = m.
            for i in range(m, n + 1):
                # candidate using x = i - m
                candidate = dp[seg - 1][i - m] - ps[i - m]
                if i == m:
                    M[i] = candidate
                else:
                    M[i] = max(M[i - 1], candidate)
            # Now fill dp[seg][i] for i from 0 to n
            for i in range(n + 1):
                if i < seg * m:
                    # Not enough elements to form seg segments.
                    dp[seg][i] = -math.inf
                else:
                    if i == 0:
                        dp[seg][i] = -math.inf  # shouldn't happen since i>= seg*m and seg>=1 implies i>=m>0
                    else:
                        # Option 1: don't form a new segment ending at i, so dp[seg][i-1]
                        option1 = dp[seg][i - 1]
                        # Option 2: form a segment ending exactly at i.
                        # We need to pick a starting point x with x <= i-m.
                        if i >= m:
                            option2 = ps[i] + M[i]
                        else:
                            option2 = -math.inf
                        dp[seg][i] = max(option1, option2)
                        
        return dp[k][n]

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums1 = [1, 2, -1, 3, 3, 4]
    k1 = 2
    m1 = 2
    print(sol.maxSum(nums1, k1, m1))  # Expected output: 13

    # Example 2:
    nums2 = [-10, 3, -1, -2]
    k2 = 4
    m2 = 1
    print(sol.maxSum(nums2, k2, m2))  # Expected output: -10