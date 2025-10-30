from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # best_min[r] will hold the minimum prefix sum we've seen so far for an index with remainder r modulo k.
        # We initialize all remainders with infinity except for remainder 0, which we set to 0 (prefix sum at index 0).
        INF = float('inf')
        best_min = [INF] * k
        best_min[0] = 0
        
        prefix = 0
        # Initialize best with a very small number.
        best = -10**18  # Sufficient lower bound given constraints (-10^9 * 2*10^5 = -2e14)
        
        # Iterate over prefix indices 1 through n.
        # prefix sum at index i corresponds to the sum of nums[0] to nums[i-1].
        for i in range(1, n + 1):
            prefix += nums[i - 1]
            mod = i % k
            # Only consider candidate if we have seen a prefix sum with the same remainder before.
            if best_min[mod] != INF:
                candidate = prefix - best_min[mod]
                best = max(best, candidate)
            # Update the minimum prefix sum for the current remainder.
            best_min[mod] = min(best_min[mod], prefix)
            
        return best

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    nums = [1, 2]
    k = 1
    print(sol.maxSubarraySum(nums, k))  # Expected output: 3

    # Example 2
    nums = [-1, -2, -3, -4, -5]
    k = 4
    print(sol.maxSubarraySum(nums, k))  # Expected output: -10

    # Example 3
    nums = [-5, 1, 2, -3, 4]
    k = 2
    print(sol.maxSubarraySum(nums, k))  # Expected output: 4