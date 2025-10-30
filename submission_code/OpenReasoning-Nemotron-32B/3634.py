from collections import defaultdict

class Solution:
	def calculateScore(self, s: str) -> int:
		stacks = defaultdict(list)
		score = 0
		for i, c in enumerate(s):
			mirror_c = chr(219 - ord(c))
			if stacks.get(mirror_c):
				j = stacks[mirror_c].pop()
				score += (i - j)
			else:
				stacks[c].append(i)
		return score