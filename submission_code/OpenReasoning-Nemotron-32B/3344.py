class Solution:
	def minimumDistance(self, points: List[List[int]]) -> int:
		n = len(points)
		u = [x + y for x, y in points]
		v = [x - y for x, y in points]
		
		u_sorted = sorted(u)
		v_sorted = sorted(v)
		
		min1_u = u_sorted[0]
		min1_u_count = 1
		idx = 1
		while idx < n and u_sorted[idx] == min1_u:
			min1_u_count += 1
			idx += 1
			
		max1_u = u_sorted[-1]
		max1_u_count = 1
		idx = n - 2
		while idx >= 0 and u_sorted[idx] == max1_u:
			max1_u_count += 1
			idx -= 1
			
		min1_v = v_sorted[0]
		min1_v_count = 1
		idx = 1
		while idx < n and v_sorted[idx] == min1_v:
			min1_v_count += 1
			idx += 1
			
		max1_v = v_sorted[-1]
		max1_v_count = 1
		idx = n - 2
		while idx >= 0 and v_sorted[idx] == max1_v:
			max1_v_count += 1
			idx -= 1
			
		candidates = set()
		for i in range(n):
			if u[i] == min1_u or u[i] == max1_u or v[i] == min1_v or v[i] == max1_v:
				candidates.add(i)
				
		ans = float('inf')
		for i in candidates:
			if u[i] == min1_u:
				if min1_u_count > 1:
					new_min_u = min1_u
				else:
					new_min_u = u_sorted[1]
			else:
				new_min_u = min1_u
				
			if u[i] == max1_u:
				if max1_u_count > 1:
					new_max_u = max1_u
				else:
					new_max_u = u_sorted[-2]
			else:
				new_max_u = max1_u
				
			if v[i] == min1_v:
				if min1_v_count > 1:
					new_min_v = min1_v
				else:
					new_min_v = v_sorted[1]
			else:
				new_min_v = min1_v
				
			if v[i] == max1_v:
				if max1_v_count > 1:
					new_max_v = max1_v
				else:
					new_max_v = v_sorted[-2]
			else:
				new_max_v = max1_v
				
			candidate_ans = max(new_max_u - new_min_u, new_max_v - new_min_v)
			if candidate_ans < ans:
				ans = candidate_ans
				
		return ans