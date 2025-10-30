class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        stack = []
        
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            
            if not stack or nums[stack[-1]] == nums[i]:
                result += i - (stack[-1] if stack else -1)
            
            stack.append(i)
        
        return result