class Solution:
	def canMakeSubsequence(self, str1: str, str2: str) -> bool:
		m = len(str2)
		i = 0
		for c in str1:
			if i == m:
				break
			if c == str2[i]:
				i += 1
			else:
				offset_c = ord(c) - ord('a')
				offset_target = ord(str2[i]) - ord('a')
				if (offset_c + 1) % 26 == offset_target:
					i += 1
		return i == m