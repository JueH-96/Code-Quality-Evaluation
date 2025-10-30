from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        # Create a prefix sum array for fast range sum queries.
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        min_sum = float('inf')
        found = False
        
        # Check for all subarray sizes between l and r
        for start in range(n):
            for size in range(l, r+1):
                end = start + size - 1
                if end >= n:
                    break
                subarray_sum = prefix[end+1] - prefix[start]
                if subarray_sum > 0:
                    found = True
                    if subarray_sum < min_sum:
                        min_sum = subarray_sum
        return min_sum if found else -1

# For quick local testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minimumSumSubarray([3, -2, 1, 4], 2, 3))  # Expected output: 1
    # Example 2:
    print(sol.minimumSumSubarray([-2, 2, -3, 1], 2, 3))  # Expected output: -1
    # Example 3:
    print(sol.minimumSumSubarray([1, 2, 3, 4], 2, 4))  # Expected output: 3