class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        left, right = [0]*n, [n]*n
        stack = []
        for i in range(n):
            while stack and nums[i] < nums[stack[-1]]:
                right[stack.pop()] = i
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        return max(right[i] - left[i] - 1 for i in range(n))