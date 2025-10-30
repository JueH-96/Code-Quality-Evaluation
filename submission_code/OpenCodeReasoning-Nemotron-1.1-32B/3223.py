class Solution:
	def countCompleteSubstrings(self, word: str, k: int) -> int:
		n = len(word)
		if n == 0:
			return 0
		
		fail = [0] * n
		for i in range(1, n):
			if abs(ord(word[i]) - ord(word[i-1])) > 2:
				fail[i] = 1
		
		next_fail = [n] * n
		last_fail = n
		for i in range(n-1, -1, -1):
			if i >= 1 and fail[i] == 1:
				last_fail = i
			next_fail[i] = last_fail
		
		pref = [[0] * 26 for _ in range(n+1)]
		for i in range(1, n+1):
			pref[i] = pref[i-1][:]
			char_index = ord(word[i-1]) - ord('a')
			pref[i][char_index] += 1
		
		count = 0
		for r in range(n):
			for d in range(1, 27):
				l_val = r - k * d + 1
				if l_val < 0:
					break
				if next_fail[l_val] > r:
					distinct_count = 0
					valid = True
					for c in range(26):
						cnt = pref[r+1][c] - pref[l_val][c]
						if cnt == 0:
							continue
						distinct_count += 1
						if cnt != k:
							valid = False
							break
					if valid and distinct_count == d:
						count += 1
		return count