class Solution:
	def findValidPair(self, s: str) -> str:
		freq = [0] * 10
		for char in s:
			d = int(char)
			freq[d] += 1
		
		for i in range(len(s) - 1):
			a = int(s[i])
			b = int(s[i+1])
			if a != b and freq[a] == a and freq[b] == b:
				return s[i] + s[i+1]
		return ""