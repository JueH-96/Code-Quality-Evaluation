class Solution:
	def maxSum(self, nums: List[int]) -> int:
		n = len(nums)
		dp = [0] * n
		sets_list = [set() for _ in range(n)]
		
		for i in range(n):
			best_val = nums[i]
			best_j = -1
			
			for j in range(i):
				if nums[i] in sets_list[j]:
					continue
				total = dp[j] + nums[i]
				if total > best_val:
					best_val = total
					best_j = j
			
			dp[i] = best_val
			if best_j == -1:
				sets_list[i] = {nums[i]}
			else:
				sets_list[i] = sets_list[best_j] | {nums[i]}
		
		return max(dp)