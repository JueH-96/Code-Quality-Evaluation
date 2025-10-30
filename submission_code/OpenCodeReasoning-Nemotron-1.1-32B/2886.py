class Solution:
	def finalString(self, s: str) -> str:
		res = []
		for char in s:
			if char == 'i':
				res.reverse()
			else:
				res.append(char)
		return ''.join(res)