class Solution:
	def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
		max_val = 100000
		divisors_list = [[] for _ in range(max_val + 1)]
		for i in range(1, max_val + 1):
			for j in range(i, max_val + 1, i):
				divisors_list[j].append(i)
		
		min_index_arr = [10**9] * (max_val + 1)
		for idx, val in enumerate(elements):
			if val <= max_val:
				if idx < min_index_arr[val]:
					min_index_arr[val] = idx
		
		res = []
		for g in groups:
			best_index = 10**9
			for d in divisors_list[g]:
				if min_index_arr[d] < best_index:
					best_index = min_index_arr[d]
			res.append(best_index if best_index != 10**9 else -1)
		
		return res