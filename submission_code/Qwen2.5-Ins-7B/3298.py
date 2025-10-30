from collections import Counter

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        count = Counter(nums)
        dp = {}
        
        def dfs(index, prev):
            if index == len(nums):
                return 0
            if (index, prev) in dp:
                return dp[(index, prev)]
            
            take = 0
            if prev == 0 or nums[index] == prev + 1 or (nums[index] == prev and count[nums[index]] > 1):
                take = 1 + dfs(index + 1, nums[index])
            skip = dfs(index + 1, prev)
            dp[(index, prev)] = max(take, skip)
            return dp[(index, prev)]
        
        return dfs(0, 0)