class Solution:
	def canMakeSubsequence(self, str1: str, str2: str) -> bool:
		j = 0
		n = len(str2)
		for i in range(len(str1)):
			if j == n:
				break
			x = ord(str1[i]) - ord('a')
			y = ord(str2[j]) - ord('a')
			if x == y or (x + 1) % 26 == y:
				j += 1
		return j == n