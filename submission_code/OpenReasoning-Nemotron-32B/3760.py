class Solution:
	def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
		if not groups:
			return []
		max_val = max(groups)
		best_candidate = [10**9] * (max_val + 1)
		
		min_index = {}
		for idx, val in enumerate(elements):
			if val not in min_index:
				min_index[val] = idx
		
		for d in range(1, max_val + 1):
			if d in min_index:
				j = min_index[d]
				x = d
				while x <= max_val:
					if j < best_candidate[x]:
						best_candidate[x] = j
					x += d
		
		res = []
		for g in groups:
			if best_candidate[g] == 10**9:
				res.append(-1)
			else:
				res.append(best_candidate[g])
		return res