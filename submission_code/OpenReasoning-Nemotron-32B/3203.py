import bisect
from typing import List

class Solution:
	def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
		n = len(s)
		total_freq = [0] * 26
		for char in s:
			idx = ord(char) - ord('a')
			total_freq[idx] += 1
		for count in total_freq:
			if count % 2 != 0:
				return [False] * len(queries)
		
		prefix = [[0] * 26 for _ in range(n+1)]
		for i in range(1, n+1):
			prefix[i] = prefix[i-1][:]
			char_idx = ord(s[i-1]) - ord('a')
			prefix[i][char_idx] += 1
		
		def get_freq(l, r):
			if l > r:
				return [0] * 26
			res = [0] * 26
			for idx in range(26):
				res[idx] = prefix[r+1][idx] - prefix[l][idx]
			return res
		
		half = n // 2
		mismatches = []
		for i in range(half):
			if s[i] != s[n-1-i]:
				mismatches.append(i)
		mismatches.sort()
		
		def has_mismatch_in_interval(l, r):
			if l > r:
				return False
			pos = bisect.bisect_left(mismatches, l)
			return pos < len(mismatches) and mismatches[pos] <= r
		
		ans = []
		for q in queries:
			a, b, c, d = q
			l1 = n - 1 - c + 1
			r1 = a - 1
			if has_mismatch_in_interval(l1, r1):
				ans.append(False)
				continue
			
			r2 = min(a, n - 1 - d) - 1
			if r2 >= 0 and has_mismatch_in_interval(0, r2):
				ans.append(False)
				continue
			
			l3 = max(b, n - 1 - c) + 1
			r3 = half - 1
			if l3 <= r3 and has_mismatch_in_interval(l3, r3):
				ans.append(False)
				continue
			
			l4 = b + 1
			r4 = n - 1 - d - 1
			if l4 <= r4 and has_mismatch_in_interval(l4, r4):
				ans.append(False)
				continue
			
			freqF = get_freq(a, b)
			freqG = get_freq(c, d)
			
			freqA = [0] * 26
			lA1 = n - 1 - b
			rA1 = min(n - 1 - a, c - 1)
			if lA1 <= rA1:
				fA1 = get_freq(lA1, rA1)
				for idx in range(26):
					freqA[idx] += fA1[idx]
			
			lA2 = max(n - 1 - b, d + 1)
			rA2 = n - 1 - a
			if lA2 <= rA2:
				fA2 = get_freq(lA2, rA2)
				for idx in range(26):
					freqA[idx] += fA2[idx]
			
			freqB = [0] * 26
			lB1 = n - 1 - d
			rB1 = min(n - 1 - c, a - 1)
			if lB1 <= rB1:
				fB1 = get_freq(lB1, rB1)
				for idx in range(26):
					freqB[idx] += fB1[idx]
			
			lB2 = max(n - 1 - d, b + 1)
			rB2 = n - 1 - c
			if lB2 <= rB2:
				fB2 = get_freq(lB2, rB2)
				for idx in range(26):
					freqB[idx] += fB2[idx]
			
			valid = True
			for idx in range(26):
				if freqF[idx] < freqA[idx] or freqG[idx] < freqB[idx]:
					valid = False
					break
				if (freqF[idx] - freqA[idx]) != (freqG[idx] - freqB[idx]):
					valid = False
					break
			ans.append(valid)
		
		return ans