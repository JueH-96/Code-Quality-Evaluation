class Solution:
	def calculateScore(self, s: str) -> int:
		stacks = [[] for _ in range(26)]
		total_score = 0
		for i, char in enumerate(s):
			idx = ord(char) - ord('a')
			mirror_idx = 25 - idx
			if stacks[mirror_idx]:
				j = stacks[mirror_idx].pop()
				total_score += (i - j)
			else:
				stacks[idx].append(i)
		return total_score