from typing import List

class Solution:
	def stringSequence(self, target: str) -> List[str]:
		if not target:
			return []
		res = ["a"]
		s = "a"
		n = len(target)
		for i in range(n):
			if i == 0:
				k = ord(target[i]) - ord('a')
				for j in range(1, k + 1):
					s = chr(ord('a') + j)
					res.append(s)
			else:
				s = s + 'a'
				res.append(s)
				k = ord(target[i]) - ord('a')
				for j in range(1, k + 1):
					s = s[:-1] + chr(ord('a') + j)
					res.append(s)
		return res