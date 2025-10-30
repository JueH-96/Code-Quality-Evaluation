from collections import defaultdict
from typing import List

class Solution:
	def shortestSubstrings(self, arr: List[str]) -> List[str]:
		n = len(arr)
		global_sub = defaultdict(set)
		for idx, s in enumerate(arr):
			n_s = len(s)
			for i in range(n_s):
				for j in range(i + 1, n_s + 1):
					substr = s[i:j]
					global_sub[substr].add(idx)
		
		res = []
		for idx, s in enumerate(arr):
			n_s = len(s)
			ans_str = ""
			for L in range(1, n_s + 1):
				candidates = set()
				for start in range(0, n_s - L + 1):
					t = s[start:start + L]
					if global_sub[t] == {idx}:
						candidates.add(t)
				if candidates:
					ans_str = min(candidates)
					break
			res.append(ans_str)
		return res