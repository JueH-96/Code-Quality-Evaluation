from collections import Counter

class Solution:
	def minGroupsForValidAssignment(self, nums: List[int]) -> int:
		n = len(nums)
		freqs = list(Counter(nums).values())
		cnt_dict = {}
		for f in freqs:
			cnt_dict[f] = cnt_dict.get(f, 0) + 1
		distinct_freqs = list(cnt_dict.keys())
		
		min_groups = n
		
		t = n
		while t >= 1:
			k = n // t
			t_min = n // (k + 1) + 1
			t_max = n // k
			
			L_val = 0
			for x in distinct_freqs:
				L_val += cnt_dict[x] * ((x + k) // (k + 1))
			
			R_val = 0
			for x in distinct_freqs:
				R_val += cnt_dict[x] * (x // k)
			
			low_bound = max(t_min, L_val)
			high_bound = min(t_max, R_val)
			
			if low_bound <= high_bound:
				min_groups = min(min_groups, low_bound)
			
			t = t_min - 1
		
		return min_groups