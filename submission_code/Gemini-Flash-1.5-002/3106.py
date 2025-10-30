class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = {}
        def solve(index, current_sum):
            if index == n:
                if current_sum == target:
                    return 0
                else:
                    return -float('inf')
            if (index, current_sum) in dp:
                return dp[(index, current_sum)]
            
            #Option 1: Include the current element
            include = solve(index + 1, current_sum + nums[index]) + 1
            
            #Option 2: Exclude the current element
            exclude = solve(index + 1, current_sum)
            
            dp[(index, current_sum)] = max(include, exclude)
            return dp[(index, current_sum)]

        result = solve(0,0)
        return result if result >=0 else -1