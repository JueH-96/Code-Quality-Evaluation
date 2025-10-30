class Solution:
	def possibleStringCount(self, word: str) -> int:
		n = len(word)
		if n == 0:
			return 0
		runs = []
		i = 0
		while i < n:
			j = i
			while j < n and word[j] == word[i]:
				j += 1
			runs.append(j - i)
			i = j
		ans = 1
		for run in runs:
			if run > 1:
				ans += run - 1
		return ans