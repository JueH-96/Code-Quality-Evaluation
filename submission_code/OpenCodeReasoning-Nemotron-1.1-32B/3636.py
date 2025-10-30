class Solution:
	def isBalanced(self, num: str) -> bool:
		total_even = 0
		total_odd = 0
		for i, digit in enumerate(num):
			if i % 2 == 0:
				total_even += int(digit)
			else:
				total_odd += int(digit)
		return total_even == total_odd