import bisect

class Solution:
	def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
		n = len(intervals)
		intervals_sorted = []
		for idx, (l, r, w) in enumerate(intervals):
			intervals_sorted.append((l, r, w, idx))
		intervals_sorted.sort(key=lambda x: (x[1], x[0], x[3]))
		
		ends = [interval[1] for interval in intervals_sorted]
		starts = [interval[0] for interval in intervals_sorted]
		
		prev_index = [-1] * n
		for i in range(n):
			lo, hi = 0, i
			pos = bisect.bisect_right(ends, starts[i], lo, hi)
			if pos > 0:
				prev_index[i] = pos - 1
		
		dp_prev = [(0, ())] * n
		best_global = (0, ())
		
		for k in range(1, 5):
			dp_curr = [(0, ())] * n
			for i in range(n):
				if i == 0:
					skip_state = (0, ())
				else:
					skip_state = dp_curr[i-1]
				
				j = prev_index[i]
				if j == -1:
					current_idx = intervals_sorted[i][3]
					take_state = (intervals_sorted[i][2], (current_idx,))
				else:
					prev_state = dp_prev[j]
					current_idx = intervals_sorted[i][3]
					new_chain = tuple(sorted(prev_state[1] + (current_idx,)))
					take_state = (prev_state[0] + intervals_sorted[i][2], new_chain)
				
				if i == 0:
					dp_curr[i] = take_state
				else:
					if skip_state[0] > take_state[0]:
						dp_curr[i] = skip_state
					elif skip_state[0] < take_state[0]:
						dp_curr[i] = take_state
					else:
						if skip_state[1] < take_state[1]:
							dp_curr[i] = skip_state
						else:
							dp_curr[i] = take_state
			
			candidate = dp_curr[n-1]
			if candidate[0] > best_global[0]:
				best_global = candidate
			elif candidate[0] == best_global[0]:
				if candidate[1] < best_global[1]:
					best_global = candidate
			
			dp_prev = dp_curr
		
		return list(best_global[1])