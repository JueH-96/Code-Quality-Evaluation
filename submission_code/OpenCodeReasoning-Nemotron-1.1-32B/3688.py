import sys
sys.setrecursionlimit(300000)

class Solution:
	def maxSubarraySum(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		
		candidate0 = -10**18
		current = 0
		for num in nums:
			current = max(num, current + num)
			candidate0 = max(candidate0, current)
		
		distinct_count = len(set(nums))
		
		if distinct_count > 300:
			if distinct_count == n:
				if n == 1:
					return candidate0
				left_dp0 = [0] * n
				left_dp0[0] = nums[0]
				for i in range(1, n):
					left_dp0[i] = max(nums[i], left_dp0[i-1] + nums[i])
				
				left_max = [0] * n
				left_max[0] = left_dp0[0]
				for i in range(1, n):
					left_max[i] = max(left_max[i-1], left_dp0[i])
				
				right_dp0 = [0] * n
				right_dp0[n-1] = nums[n-1]
				for i in range(n-2, -1, -1):
					right_dp0[i] = max(nums[i], right_dp0[i+1] + nums[i])
				
				right_max = [0] * n
				right_max[n-1] = right_dp0[n-1]
				for i in range(n-2, -1, -1):
					right_max[i] = max(right_max[i+1], right_dp0[i])
				
				candidate_for_index = [-10**18] * n
				for i in range(n):
					if i == 0:
						candidate_for_index[i] = right_max[1] if n > 1 else -10**18
					elif i == n-1:
						candidate_for_index[i] = left_max[n-2]
					else:
						cross = left_dp0[i-1] + right_dp0[i+1]
						candidate_for_index[i] = max(left_max[i-1], right_max[i+1], cross)
				
				candidate_x = max(candidate_for_index)
				ans = max(candidate0, candidate_x)
				return ans
			else:
				ans = candidate0
				unique_nums = set(nums)
				for x in unique_nums:
					count_x = nums.count(x)
					if count_x == n:
						continue
					current_val = 0
					best_candidate = -10**18
					for num in nums:
						if num == x:
							current_val = 0
						else:
							current_val = max(num, current_val + num)
							best_candidate = max(best_candidate, current_val)
					ans = max(ans, best_candidate)
				return ans
		else:
			ans = candidate0
			unique_nums = set(nums)
			for x in unique_nums:
				count_x = nums.count(x)
				if count_x == n:
					continue
				current_val = 0
				best_candidate = -10**18
				for num in nums:
					if num == x:
						current_val = 0
					else:
						current_val = max(num, current_val + num)
						best_candidate = max(best_candidate, current_val)
				ans = max(ans, best_candidate)
			return ans