class Solution:
	def smallestString(self, s: str) -> str:
		n = len(s)
		s_list = list(s)
		i = 0
		while i < n and s_list[i] == 'a':
			i += 1
		if i == n:
			s_list[-1] = 'z'
			return ''.join(s_list)
		j = i
		while j < n and s_list[j] != 'a':
			j += 1
		for k in range(i, j):
			c = s_list[k]
			new_char = chr((ord(c) - ord('a') - 1) % 26 + ord('a'))
			s_list[k] = new_char
		return ''.join(s_list)