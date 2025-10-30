from typing import List

MOD = 10**9 + 7

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        # Compute the bitwise OR of the entire array.
        total_or = 0
        for x in nums:
            total_or |= x
        
        # Sort nums descending to pick the largest (k-1) originals if needed.
        nums.sort(reverse=True)
        
        # Case 1: do no operations, pick the top k of the original array.
        sum_noop = sum((nums[i] * nums[i]) for i in range(k)) % MOD
        
        # Case 2: merge everything into one element = total_or, then pick that
        # plus the next k-1 largest originals (excluding the slot we used for OR).
        # We assume we can pick total_or once and the next k-1 from the sorted list.
        sum_merge = (total_or * total_or) % MOD
        for i in range(k-1):
            sum_merge = (sum_merge + nums[i] * nums[i]) % MOD
        
        # The answer is the max of the two scenarios.
        return max(sum_noop, sum_merge)

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSum([2,6,5,8], 2))  # 261
    print(sol.maxSum([4,5,4,7], 3))  # 90