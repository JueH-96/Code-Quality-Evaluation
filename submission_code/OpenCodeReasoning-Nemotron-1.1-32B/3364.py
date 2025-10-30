from typing import List

class Solution:
	def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
		n = len(nums)
		m = len(andValues)
		INF = 10**18
		dp_prev = [INF] * (n + 1)
		dp_prev[0] = 0
		
		for j in range(m):
			dp_cur = [INF] * (n + 1)
			cur_dict = {}
			for i in range(n):
				cost_prev = dp_prev[i]
				
				new_dict = {}
				if nums[i] in new_dict:
					if cost_prev < new_dict[nums[i]]:
						new_dict[nums[i]] = cost_prev
				else:
					new_dict[nums[i]] = cost_prev
				
				if i > 0:
					for and_val, min_val in cur_dict.items():
						new_and = and_val & nums[i]
						if new_and in new_dict:
							if min_val < new_dict[new_and]:
								new_dict[new_and] = min_val
						else:
							new_dict[new_and] = min_val
				
				if andValues[j] in new_dict:
					candidate = new_dict[andValues[j]] + nums[i]
					if candidate < dp_cur[i + 1]:
						dp_cur[i + 1] = candidate
				
				cur_dict = new_dict
			
			dp_prev = dp_cur
		
		return dp_cur[n] if dp_cur[n] < INF else -1