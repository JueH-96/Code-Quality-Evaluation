from collections import defaultdict

class Solution:
	def maximumLength(self, nums: List[int], k: int) -> int:
		n = len(nums)
		if n == 0:
			return 0
		
		best_same_dict = defaultdict(lambda: [0] * (k+1))
		best1 = [0] * (k+1)
		best2 = [0] * (k+1)
		best1_value = [None] * (k+1)
		
		ans = 0
		
		for i in range(n):
			v = nums[i]
			dp_i = [1] * (k+1)
			
			for c in range(k+1):
				candidate = best_same_dict[v][c] + 1
				if candidate > dp_i[c]:
					dp_i[c] = candidate
			
			for c in range(k):
				if best1_value[c] != v:
					candidate = best1[c] + 1
				else:
					candidate = best2[c] + 1
				if candidate > dp_i[c+1]:
					dp_i[c+1] = candidate
			
			for c in range(k+1):
				if dp_i[c] > best_same_dict[v][c]:
					best_same_dict[v][c] = dp_i[c]
			
			for c in range(k+1):
				x = dp_i[c]
				if x > best1[c]:
					if best1_value[c] == v:
						best1[c] = x
					else:
						best2[c] = best1[c]
						best1[c] = x
						best1_value[c] = v
				else:
					if best1_value[c] != v:
						if x > best2[c]:
							best2[c] = x
			
			for c in range(k+1):
				if dp_i[c] > ans:
					ans = dp_i[c]
		
		return ans