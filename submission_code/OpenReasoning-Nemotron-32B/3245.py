class Solution:
	def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
		n = len(s)
		len_a = len(a)
		len_b = len(b)
		a_indices = []
		for i in range(n - len_a + 1):
			if s[i:i+len_a] == a:
				a_indices.append(i)
				
		b_indices = []
		for i in range(n - len_b + 1):
			if s[i:i+len_b] == b:
				b_indices.append(i)
				
		j = 0
		res = []
		for i in a_indices:
			low_bound = i - k
			high_bound = i + k
			while j < len(b_indices) and b_indices[j] < low_bound:
				j += 1
			if j < len(b_indices) and b_indices[j] <= high_bound:
				res.append(i)
				
		return res