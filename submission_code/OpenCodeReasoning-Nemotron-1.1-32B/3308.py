class Solution:
	def lastNonEmptyString(self, s: str) -> str:
		n = len(s)
		count = [0] * 26
		for char in s:
			count[ord(char) - ord('a')] += 1
		
		max_freq = max(count)
		
		max_letters = set()
		for i in range(26):
			if count[i] == max_freq:
				max_letters.add(chr(ord('a') + i))
		
		res = []
		for i in range(n-1, -1, -1):
			if not max_letters:
				break
			char = s[i]
			if char in max_letters:
				res.append(char)
				max_letters.remove(char)
		
		return ''.join(res[::-1])