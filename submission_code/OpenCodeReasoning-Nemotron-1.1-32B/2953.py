class Solution:
	def countPairs(self, coordinates: List[List[int]], k: int) -> int:
		total = 0
		freq = {}
		for x, y in coordinates:
			for a in range(0, k + 1):
				b = k - a
				candidate = (x ^ a, y ^ b)
				total += freq.get(candidate, 0)
			freq[(x, y)] = freq.get((x, y), 0) + 1
		return total