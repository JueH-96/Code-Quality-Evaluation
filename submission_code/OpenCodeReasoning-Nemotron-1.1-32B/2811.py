class Solution:
	def minimumSum(self, n: int, k: int) -> int:
		chosen = set()
		forbidden = set()
		x = 1
		while len(chosen) < n:
			if x not in forbidden:
				chosen.add(x)
				comp = k - x
				if comp > 0:
					forbidden.add(comp)
			x += 1
		return sum(chosen)