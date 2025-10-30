class Solution:
	def doesValidArrayExist(self, derived: List[int]) -> bool:
		total = 0
		for num in derived:
			total ^= num
		return total == 0