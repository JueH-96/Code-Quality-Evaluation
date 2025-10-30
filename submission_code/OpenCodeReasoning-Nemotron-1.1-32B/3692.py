import bisect

def kmp_search(text, pattern):
	if pattern == "":
		return list(range(len(text) + 1))
	n = len(text)
	m = len(pattern)
	lps = [0] * m
	length = 0
	i = 1
	while i < m:
		if pattern[i] == pattern[length]:
			length += 1
			lps[i] = length
			i += 1
		else:
			if length != 0:
				length = lps[length - 1]
			else:
				lps[i] = 0
				i += 1
	i = 0
	j = 0
	res = []
	while i < n:
		if j < m and text[i] == pattern[j]:
			i += 1
			j += 1
		if j == m:
			res.append(i - j)
			j = lps[j - 1] if j > 0 else 0
		elif i < n and text[i] != pattern[j]:
			if j != 0:
				j = lps[j - 1]
			else:
				i += 1
	return res

class Solution:
	def shortestMatchingSubstring(self, s: str, p: str) -> int:
		star_index1 = p.find('*')
		star_index2 = p.find('*', star_index1 + 1)
		p1 = p[:star_index1]
		p2 = p[star_index1 + 1:star_index2]
		p3 = p[star_index2 + 1:]
		L1, L2, L3 = len(p1), len(p2), len(p3)
		
		if L1 == 0:
			A = list(range(len(s) + 1))
		else:
			A = kmp_search(s, p1)
		
		if L2 == 0:
			B = list(range(len(s) + 1))
		else:
			B = kmp_search(s, p2)
		
		if L3 == 0:
			C = None
		else:
			C = kmp_search(s, p3)
		
		A.sort()
		if C is not None:
			C.sort()
		
		ans = float('inf')
		
		for k in B:
			if L1 > 0:
				pos = bisect.bisect_right(A, k - L1) - 1
				if pos < 0:
					continue
				i_max = A[pos]
			else:
				pos = bisect.bisect_right(A, k) - 1
				if pos < 0:
					continue
				i_max = A[pos]
			
			if L3 > 0:
				pos = bisect.bisect_left(C, k + L2)
				if pos >= len(C):
					continue
				j0_min = C[pos]
				j_end = j0_min + L3 - 1
				length = j_end - i_max + 1
			else:
				j_end = k + L2 - 1
				length = j_end - i_max + 1
			
			if length < ans:
				ans = length
		
		return ans if ans != float('inf') else -1