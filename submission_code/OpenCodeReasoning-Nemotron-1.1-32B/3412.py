class Solution:
	def findPermutationDifference(self, s: str, t: str) -> int:
		s_dict = {char: idx for idx, char in enumerate(s)}
		total = 0
		for idx, char in enumerate(t):
			total += abs(s_dict[char] - idx)
		return total