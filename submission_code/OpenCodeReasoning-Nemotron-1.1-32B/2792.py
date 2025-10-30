class Solution:
	def doesValidArrayExist(self, derived: List[int]) -> bool:
		total = 0
		for bit in derived:
			total ^= bit
		return total == 0