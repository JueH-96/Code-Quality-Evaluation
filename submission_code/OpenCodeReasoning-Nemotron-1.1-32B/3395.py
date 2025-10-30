import math
from collections import Counter

class Solution:
	def minAnagramLength(self, s: str) -> int:
		n = len(s)
		cnt = Counter(s)
		g = 0
		for count in cnt.values():
			g = math.gcd(g, count)
		m = math.gcd(n, g)
		return n // m