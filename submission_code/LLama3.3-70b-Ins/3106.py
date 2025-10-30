from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        memo = {}
        
        def dfs(i, target):
            if (i, target) in memo:
                return memo[(i, target)]
            if target == 0:
                return 0
            if i == len(nums) or target < 0:
                return float('-inf')
            
            # include current element
            include = 1 + dfs(i + 1, target - nums[i])
            
            # exclude current element
            exclude = dfs(i + 1, target)
            
            memo[(i, target)] = max(include, exclude)
            return memo[(i, target)]
        
        result = dfs(0, target)
        return result if result != float('-inf') else -1