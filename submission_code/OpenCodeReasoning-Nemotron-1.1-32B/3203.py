from typing import List

class Solution:
	def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
		n = len(s)
		total_freq = [0] * 26
		for char in s:
			total_freq[ord(char) - ord('a')] += 1
		
		global_odd = False
		for count in total_freq:
			if count % 2 != 0:
				global_odd = True
				break
		
		if global_odd:
			return [False] * len(queries)
		
		L = n // 2
		match = [1] * L
		for i in range(L):
			if s[i] != s[n-1-i]:
				match[i] = 0
		
		prefix0 = [0] * (L+1)
		for i in range(1, L+1):
			prefix0[i] = prefix0[i-1] + (1 - match[i-1])
		
		ans = []
		for query in queries:
			a, b, c, d = query
			flag1 = True
			if a > 0:
				r1 = min(a-1, n-d-2)
				if r1 >= 0:
					if prefix0[r1+1] - prefix0[0] > 0:
						flag1 = False
			if flag1 and b < L-1:
				r2 = min(L-1, n-d-2)
				if r2 >= b+1:
					if prefix0[r2+1] - prefix0[b+1] > 0:
						flag1 = False
			
			flag2 = True
			if n - c <= L - 1:
				if a > 0:
					l1 = n - c
					if l1 <= a-1:
						if prefix0[a] - prefix0[l1] > 0:
							flag2 = False
				if flag2 and b < L-1:
					l2 = max(b+1, n-c)
					if l2 <= L-1:
						if prefix0[L] - prefix0[l2] > 0:
							flag2 = False
			
			if flag1 and flag2:
				ans.append(True)
			else:
				ans.append(False)
		
		return ans