from typing import List

class Solution:
	def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
		n = len(s)
		len_a = len(a)
		len_b = len(b)
		
		indices_a = []
		for i in range(n - len_a + 1):
			if s[i:i+len_a] == a:
				indices_a.append(i)
				
		indices_b = []
		for i in range(n - len_b + 1):
			if s[i:i+len_b] == b:
				indices_b.append(i)
				
		if not indices_b:
			return []
		
		j = 0
		res = []
		for i in indices_a:
			while j < len(indices_b) and indices_b[j] < i - k:
				j += 1
			if j < len(indices_b) and indices_b[j] <= i + k:
				res.append(i)
				
		return res