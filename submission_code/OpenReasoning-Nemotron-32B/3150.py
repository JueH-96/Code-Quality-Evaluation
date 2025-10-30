class Solution:
	def shortestBeautifulSubstring(self, s: str, k: int) -> str:
		min_len = float('inf')
		candidate = ""
		n = len(s)
		
		for i in range(n):
			count = 0
			for j in range(i, n):
				if s[j] == '1':
					count += 1
				if count > k:
					break
				if count == k:
					length = j - i + 1
					if length < min_len:
						min_len = length
						candidate = s[i:j+1]
					elif length == min_len:
						if s[i:j+1] < candidate:
							candidate = s[i:j+1]
					break
		
		return candidate if min_len != float('inf') else ""