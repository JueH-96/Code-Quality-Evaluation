class Solution:
	def minimumString(self, a: str, b: str, c: str) -> str:
		def merge(s1, s2):
			if s2 in s1:
				return s1
			n = min(len(s1), len(s2))
			for i in range(n, 0, -1):
				if s1.endswith(s2[:i]):
					return s1 + s2[i:]
			return s1 + s2
		
		orders = [
			(a, b, c),
			(a, c, b),
			(b, a, c),
			(b, c, a),
			(c, a, b),
			(c, b, a)
		]
		
		best = None
		for order in orders:
			merged1 = merge(order[0], order[1])
			merged2 = merge(merged1, order[2])
			if best is None:
				best = merged2
			else:
				if len(merged2) < len(best):
					best = merged2
				elif len(merged2) == len(best) and merged2 < best:
					best = merged2
		return best