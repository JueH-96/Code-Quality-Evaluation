from typing import List

class Solution:
	def stringSequence(self, target: str) -> List[str]:
		res = []
		s = ""
		for char in target:
			steps = ord(char) - ord('a')
			for i in range(steps + 1):
				res.append(s + chr(ord('a') + i))
			s += char
		return res