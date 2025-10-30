class Solution:
	def distributeCandies(self, n: int, limit: int) -> int:
		def comb2(x):
			if x < 2:
				return 0
			return x * (x - 1) // 2
		
		total = comb2(n + 2)
		total -= 3 * comb2(n - limit + 1)
		total += 3 * comb2(n - 2 * limit)
		total -= comb2(n - 3 * limit - 1)
		return total