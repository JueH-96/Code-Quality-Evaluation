class Solution:
	def finalString(self, s: str) -> str:
		res = []
		for c in s:
			if c == 'i':
				res.reverse()
			else:
				res.append(c)
		return ''.join(res)