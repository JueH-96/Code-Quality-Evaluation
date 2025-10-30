class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        left, right = [0]*n, [0]*n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                right[stack.pop()] = i
            left[i] = stack[-1] if stack else -1
            stack.append(i)
        while stack:
            right[stack.pop()] = n
        return sum((i-left[i])*(right[i]-i) for i in range(n))