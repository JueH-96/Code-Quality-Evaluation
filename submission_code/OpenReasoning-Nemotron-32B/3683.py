class Solution:
	def answerString(self, word: str, numFriends: int) -> str:
		n = len(word)
		best_start = -1
		best_end = -1
		for i in range(n):
			j_max = n - (numFriends - 1) + min(i, numFriends - 1)
			if best_start == -1:
				best_start, best_end = i, j_max
			else:
				len_candidate = j_max - i
				len_best = best_end - best_start
				for k in range(min(len_candidate, len_best)):
					if word[i + k] > word[best_start + k]:
						best_start, best_end = i, j_max
						break
					elif word[i + k] < word[best_start + k]:
						break
				else:
					if len_candidate > len_best:
						best_start, best_end = i, j_max
		return word[best_start:best_end]