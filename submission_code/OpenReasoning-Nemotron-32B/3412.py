class Solution:
	def findPermutationDifference(self, s: str, t: str) -> int:
		t_map = {}
		for idx, char in enumerate(t):
			t_map[char] = idx
		
		total_diff = 0
		for idx, char in enumerate(s):
			total_diff += abs(idx - t_map[char])
		
		return total_diff