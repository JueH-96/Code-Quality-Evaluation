class Solution:
	def smallestString(self, s: str) -> str:
		n = len(s)
		res = []
		i = 0
		while i < n and s[i] == 'a':
			res.append(s[i])
			i += 1
		
		if i == n:
			return s[:-1] + 'z'
		
		j = i
		while j < n and s[j] != 'a':
			j += 1
		
		for k in range(i, j):
			res.append(chr(ord(s[k]) - 1))
		
		for k in range(j, n):
			res.append(s[k])
		
		return ''.join(res)