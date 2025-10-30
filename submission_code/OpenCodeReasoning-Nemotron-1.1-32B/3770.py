class Solution:
	def generateString(self, str1: str, str2: str) -> str:
		n = len(str1)
		m = len(str2)
		L = n + m - 1
		ans = [None] * L
		
		for i in range(n):
			if str1[i] == 'T':
				for j in range(m):
					pos = i + j
					if pos >= L:
						break
					if ans[pos] is not None and ans[pos] != str2[j]:
						return ""
					ans[pos] = str2[j]
		
		constraint_broken = [False] * n
		for j in range(n):
			if str1[j] == 'F':
				for k in range(j, j + m):
					if k >= L:
						break
					if ans[k] is not None:
						if ans[k] != str2[k - j]:
							constraint_broken[j] = True
							break
		
		for i in range(L):
			if ans[i] is not None:
				continue
				
			start_j = max(0, i - m + 1)
			end_j = min(n - 1, i)
			
			found = False
			for c in "abcdefghijklmnopqrstuvwxyz":
				valid = True
				for j in range(start_j, end_j + 1):
					if str1[j] != 'F':
						continue
					if constraint_broken[j]:
						continue
					offset = i - j
					if offset == m - 1:
						if c == str2[offset]:
							valid = False
							break
				if not valid:
					continue
					
				ans[i] = c
				found = True
				for j in range(start_j, end_j + 1):
					if str1[j] != 'F':
						continue
					if constraint_broken[j]:
						continue
					offset = i - j
					if c != str2[offset]:
						constraint_broken[j] = True
				break
				
			if not found:
				return ""
				
		return ''.join(ans)