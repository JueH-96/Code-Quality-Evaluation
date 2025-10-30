from collections import defaultdict
from typing import List

class Solution:
	def countPairs(self, coordinates: List[List[int]], k: int) -> int:
		freq = defaultdict(int)
		for coord in coordinates:
			freq[tuple(coord)] += 1
		
		total = 0
		for (x, y) in freq:
			for a in range(0, k + 1):
				b = k - a
				x2 = x ^ a
				y2 = y ^ b
				target = (x2, y2)
				if target in freq:
					if target == (x, y):
						c = freq[(x, y)]
						total += c * (c - 1) // 2
					else:
						if (x, y) < target:
							total += freq[(x, y)] * freq[target]
		return total