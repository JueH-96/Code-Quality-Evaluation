class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        left = [0]*n
        right = [0]*n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                right[stack.pop()] = i
            stack.append(i)
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                left[stack.pop()] = i
            stack.append(i)
        res = 0
        for i in range(n):
            res += nums[i]**2*(i-left[i]+1)*(right[i]-i+1)
            res %= mod
        return res