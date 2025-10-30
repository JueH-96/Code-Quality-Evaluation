class Solution:
	def transformArray(self, nums: List[int]) -> List[int]:
		arr = [0 if num % 2 == 0 else 1 for num in nums]
		arr.sort()
		return arr