import bisect

class Solution:
	def shortestMatchingSubstring(self, s: str, p: str) -> int:
		parts = p.split('*')
		if len(parts) < 3:
			parts += [''] * (3 - len(parts))
		else:
			parts = parts[:3]
		prefix, middle, suffix = parts
		
		list1 = self.kmp(s, prefix)
		list2 = self.kmp(s, middle) if middle != "" else None
		list3 = self.kmp(s, suffix)
		
		ans = 10**9
		
		if middle != "":
			for i in list1:
				j_idx = bisect.bisect_left(list2, i + len(prefix))
				if j_idx == len(list2):
					continue
				j = list2[j_idx]
				k_idx = bisect.bisect_left(list3, j + len(middle))
				if k_idx == len(list3):
					continue
				k = list3[k_idx]
				length = k + len(suffix) - i
				if length < ans:
					ans = length
		else:
			for i in list1:
				j = i + len(prefix)
				k_idx = bisect.bisect_left(list3, j)
				if k_idx == len(list3):
					continue
				k = list3[k_idx]
				length = k + len(suffix) - i
				if length < ans:
					ans = length
		
		return ans if ans != 10**9 else -1
	
	def kmp(self, s, pat):
		if pat == "":
			return list(range(len(s) + 1))
		n = len(s)
		m = len(pat)
		pi = [0] * m
		k = 0
		for q in range(1, m):
			while k > 0 and pat[k] != pat[q]:
				k = pi[k-1]
			if pat[k] == pat[q]:
				k += 1
			else:
				k = 0
			pi[q] = k
		
		indices = []
		q_val = 0
		for i in range(n):
			while q_val > 0 and s[i] != pat[q_val]:
				q_val = pi[q_val-1]
			if s[i] == pat[q_val]:
				q_val += 1
			else:
				q_val = 0
			if q_val == m:
				start_index = i - m + 1
				indices.append(start_index)
				q_val = pi[q_val-1]
		return indices