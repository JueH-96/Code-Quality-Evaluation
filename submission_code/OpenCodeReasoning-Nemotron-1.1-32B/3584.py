import bisect
from collections import defaultdict
from typing import List

class Solution:
	def validSequence(self, word1: str, word2: str) -> List[int]:
		n = len(word1)
		m = len(word2)
		if m > n:
			return []
		
		indices_dict = defaultdict(list)
		for idx, char in enumerate(word1):
			indices_dict[char].append(idx)
		
		all_indices = list(range(n))
		
		dp0 = [None] * (m + 1)
		dp1 = [None] * (m + 1)
		
		dp0[0] = (-1, -1)
		dp1[0] = (-1, -1)
		
		for i in range(m):
			if dp0[i] is not None:
				last_index0 = dp0[i][0]
				c = word2[i]
				if c in indices_dict:
					lst = indices_dict[c]
					pos = bisect.bisect_right(lst, last_index0)
					if pos < len(lst):
						j = lst[pos]
						if dp0[i + 1] is None or j < dp0[i + 1][0]:
							dp0[i + 1] = (j, 0)
				pos = bisect.bisect_right(all_indices, last_index0)
				if pos < len(all_indices):
					j = all_indices[pos]
					if dp1[i + 1] is None or j < dp1[i + 1][0]:
						dp1[i + 1] = (j, 0)
			
			if dp1[i] is not None:
				last_index1 = dp1[i][0]
				c = word2[i]
				if c in indices_dict:
					lst = indices_dict[c]
					pos = bisect.bisect_right(lst, last_index1)
					if pos < len(lst):
						j = lst[pos]
						if dp1[i + 1] is None or j < dp1[i + 1][0]:
							dp1[i + 1] = (j, 1)
		
		def recover(dp0, dp1, state_type, m):
			seq = []
			current_state = state_type
			for i in range(m, 0, -1):
				if current_state == 0:
					entry = dp0[i]
				else:
					entry = dp1[i]
				seq.append(entry[0])
				current_state = entry[1]
			seq.reverse()
			return seq
		
		res0 = recover(dp0, dp1, 0, m) if dp0[m] is not None else None
		res1 = recover(dp0, dp1, 1, m) if dp1[m] is not None else None
		
		if res0 is not None and res1 is not None:
			return res0 if res0 < res1 else res1
		elif res0 is not None:
			return res0
		elif res1 is not None:
			return res1
		else:
			return []