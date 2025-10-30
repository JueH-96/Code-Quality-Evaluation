class Solution:
	def getSmallestString(self, s: str, k: int) -> str:
		res = []
		for char in s:
			for c in "abcdefghijklmnopqrstuvwxyz":
				diff = abs(ord(char) - ord(c))
				cost = min(diff, 26 - diff)
				if cost <= k:
					res.append(c)
					k -= cost
					break
		return ''.join(res)