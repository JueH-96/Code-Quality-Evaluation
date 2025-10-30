import bisect
from typing import List

class Solution:
	def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
		n = len(intervals)
		if n == 0:
			return []
		
		intervals_sorted = sorted([(l, r, w, i) for i, (l, r, w) in enumerate(intervals)], key=lambda x: x[1])
		
		r_list = [interv[1] for interv in intervals_sorted]
		last = []
		for i in range(n):
			l_i = intervals_sorted[i][0]
			j0 = bisect.bisect_left(r_list, l_i)
			last.append(j0 - 1)
		
		best0 = [(0, tuple())] * n
		best_prev = best0
		best_all = [best0]
		
		for k_val in range(1, 5):
			current_best = [(0, tuple()) for _ in range(n)]
			for i in range(n):
				if i == 0:
					candidate1 = (0, tuple())
				else:
					candidate1 = current_best[i-1]
				
				j = last[i]
				if j == -1:
					prev_state = (0, tuple())
				else:
					prev_state = best_prev[j]
				
				w_i = intervals_sorted[i][2]
				orig_idx_i = intervals_sorted[i][3]
				new_weight = prev_state[0] + w_i
				new_list = list(prev_state[1])
				new_list.append(orig_idx_i)
				new_list.sort()
				new_tuple = tuple(new_list)
				candidate2 = (new_weight, new_tuple)
				
				if candidate1[0] > candidate2[0]:
					chosen = candidate1
				elif candidate1[0] < candidate2[0]:
					chosen = candidate2
				else:
					if candidate1[1] < candidate2[1]:
						chosen = candidate1
					else:
						chosen = candidate2
				
				current_best[i] = chosen
			
			best_all.append(current_best)
			best_prev = current_best
		
		candidates = [best_all[k][n-1] for k in range(5)]
		result_candidate = candidates[0]
		for i in range(1, 5):
			cand = candidates[i]
			if cand[0] > result_candidate[0]:
				result_candidate = cand
			elif cand[0] == result_candidate[0]:
				if cand[1] < result_candidate[1]:
					result_candidate = cand
		
		return list(result_candidate[1])