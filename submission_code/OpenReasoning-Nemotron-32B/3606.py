class Solution:
	def minElement(self, nums: List[int]) -> int:
		min_val = float('inf')
		for num in nums:
			s = 0
			for digit in str(num):
				s += int(digit)
			if s < min_val:
				min_val = s
		return min_val