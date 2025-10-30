class Solution:
	def maxOperations(self, s: str) -> int:
		s = list(s)
		n = len(s)
		ans = 0
		i = 0
		while i < n:
			if s[i] == '1':
				if i + 1 < n and s[i + 1] == '0':
					j = i + 1
					while j < n and s[j] == '0':
						j += 1
					ans += 1
					s[i] = '0'
					if j < n:
						s[j - 1] = '1'
					else:
						s[n - 1] = '1'
					if i > 0:
						i -= 1
					else:
						i += 1
				else:
					i += 1
			else:
				i += 1
		return ans