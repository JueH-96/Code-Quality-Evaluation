class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = {}

        def solve(index, current_sum):
            if (index, current_sum) in dp:
                return dp[(index, current_sum)]

            if current_sum == target:
                return 0
            
            if current_sum > target or index == n:
                return -float('inf')

            include = 1 + solve(index + 1, current_sum + nums[index])
            exclude = solve(index + 1, current_sum)

            dp[(index, current_sum)] = max(include, exclude)
            return dp[(index, current_sum)]

        result = solve(0, 0)
        return result if result != -float('inf') else -1