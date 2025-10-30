class Solution:
	def maxArrayValue(self, nums: List[int]) -> int:
		stack = []
		for num in reversed(nums):
			while stack and num <= stack[-1]:
				num += stack.pop()
			stack.append(num)
		return max(stack)