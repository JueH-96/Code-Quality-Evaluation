class Solution:
	def makeSmallestPalindrome(self, s: str) -> str:
		n = len(s)
		s_list = list(s)
		for i in range(n // 2):
			j = n - 1 - i
			if s_list[i] != s_list[j]:
				if s_list[i] < s_list[j]:
					s_list[j] = s_list[i]
				else:
					s_list[i] = s_list[j]
		return ''.join(s_list)