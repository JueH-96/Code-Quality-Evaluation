class Solution:
	def minimumDistance(self, points: List[List[int]]) -> int:
		n = len(points)
		u = [x + y for x, y in points]
		v = [x - y for x, y in points]
		
		base = max(max(u) - min(u), max(v) - min(v))
		
		sorted_u = sorted(u)
		sorted_v = sorted(v)
		
		min1_u = sorted_u[0]
		min2_u = sorted_u[1]
		max1_u = sorted_u[-1]
		max2_u = sorted_u[-2]
		
		min1_v = sorted_v[0]
		min2_v = sorted_v[1]
		max1_v = sorted_v[-1]
		max2_v = sorted_v[-2]
		
		candidate_indices = set()
		for i in range(n):
			if u[i] == min1_u or u[i] == max1_u or v[i] == min1_v or v[i] == max1_v:
				candidate_indices.add(i)
		
		ans = base
		for i in candidate_indices:
			if u[i] == min1_u:
				new_min_u = min2_u
			else:
				new_min_u = min1_u
				
			if u[i] == max1_u:
				new_max_u = max2_u
			else:
				new_max_u = max1_u
				
			if v[i] == min1_v:
				new_min_v = min2_v
			else:
				new_min_v = min1_v
				
			if v[i] == max1_v:
				new_max_v = max2_v
			else:
				new_max_v = max1_v
				
			candidate_ans = max(new_max_u - new_min_u, new_max_v - new_min_v)
			if candidate_ans < ans:
				ans = candidate_ans
		
		return ans