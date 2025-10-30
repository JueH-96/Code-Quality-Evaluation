from collections import defaultdict
from typing import List

class Solution:
	def shortestSubstrings(self, arr: List[str]) -> List[str]:
		n = len(arr)
		if n == 0:
			return []
		freq = defaultdict(int)
		sets = []
		for s in arr:
			subs = set()
			len_s = len(s)
			for i in range(len_s):
				for j in range(i+1, len_s+1):
					subs.add(s[i:j])
			sets.append(subs)
			for sub in subs:
				freq[sub] += 1
		
		res = []
		for i in range(n):
			s = arr[i]
			len_s = len(s)
			found = False
			ans_str = ""
			for L in range(1, len_s+1):
				candidates = [sub for sub in sets[i] if len(sub) == L]
				candidates.sort()
				for cand in candidates:
					if freq[cand] == 1:
						ans_str = cand
						found = True
						break
				if found:
					break
			res.append(ans_str)
		return res