class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        from functools import lru_cache

        @lru_cache(None)
        def dfs(index, current_sum):
            if current_sum == target:
                return 0  # Return 0 to start counting the length of the subsequence
            if current_sum > target or index == len(nums):
                return float('-inf')  # Impossible case, return negative infinity

            # Option 1: Skip the current element
            skip = dfs(index + 1, current_sum)

            # Option 2: Include the current element and increase the count by 1
            take = 1 + dfs(index + 1, current_sum + nums[index])

            return max(skip, take)

        result = dfs(0, 0)
        return result if result != float('-inf') else -1