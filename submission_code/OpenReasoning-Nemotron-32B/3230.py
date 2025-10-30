class Solution:
	def removeAlmostEqualCharacters(self, word: str) -> int:
		base = ord('a')
		n = len(word)
		dp_prev = [0] * 26
		target0 = ord(word[0]) - base
		for c in range(26):
			if c == target0:
				dp_prev[c] = 0
			else:
				dp_prev[c] = 1
		
		for i in range(1, n):
			dp_curr = [10**9] * 26
			target = ord(word[i]) - base
			for c in range(26):
				cost_here = 0 if c == target else 1
				for p in range(26):
					if abs(c - p) <= 1:
						continue
					if dp_prev[p] + cost_here < dp_curr[c]:
						dp_curr[c] = dp_prev[p] + cost_here
			dp_prev = dp_curr
		
		return min(dp_prev)