from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # For each index j, dp[j] is a bitmask where the i-th bit
        # is 1 if and only if we can pick a subset of queries (affecting index j)
        # that sum exactly to i.
        # Initially, only sum 0 is achievable.
        dp = [1] * n
        # For each index j, we only need to record sums up to nums[j].
        masks = [(1 << (nums[j] + 1)) - 1 for j in range(n)]
        
        # If nums is already a Zero Array, no queries are needed.
        if all(num == 0 for num in nums):
            return 0
        
        # Process the queries one by one.
        # Each query (l, r, val) gives us the option to subtract 'val' 
        # from any chosen indices in the segment [l, r].
        # In our dp approach, that means for each index j in [l, r],
        # we can update the achievable sums via a bit-shift.
        for k, (l, r, val) in enumerate(queries):
            for j in range(l, r + 1):
                dp[j] |= (dp[j] << val) & masks[j]
            # After processing these queries, check if every index j 
            # can have its exact initial value reduced to zero.
            # That is, we want the bit representing nums[j] to be set in dp[j].
            if all((dp[j] >> nums[j]) & 1 for j in range(n)):
                return k + 1
        return -1

# Below are some tests you can run locally:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    nums = [2, 0, 2]
    queries = [[0,2,1], [0,2,1], [1,1,3]]
    print(sol.minZeroArray(nums, queries))  # Expected output: 2

    # Example 2
    nums = [4, 3, 2, 1]
    queries = [[1,3,2], [0,2,1]]
    print(sol.minZeroArray(nums, queries))  # Expected output: -1

    # Example 3
    nums = [1,2,3,2,1]
    queries = [[0,1,1], [1,2,1], [2,3,2], [3,4,1], [4,4,1]]
    print(sol.minZeroArray(nums, queries))  # Expected output: 4

    # Example 4
    nums = [1,2,3,2,6]
    queries = [[0,1,1], [0,2,1], [1,4,2], [4,4,4], [3,4,1], [4,4,5]]
    print(sol.minZeroArray(nums, queries))  # Expected output: 4