class Solution:
	def maximumSubarraySum(self, nums: List[int], k: int) -> int:
		n = len(nums)
		if n < 2:
			return 0
		P = [0] * (n + 1)
		for i in range(1, n + 1):
			P[i] = P[i - 1] + nums[i - 1]
		
		min_prefix = {nums[0]: 0}
		ans = -10**18
		
		for j in range(1, n):
			candidate = -10**18
			target1 = nums[j] + k
			target2 = nums[j] - k
			
			if target1 in min_prefix:
				candidate = max(candidate, P[j + 1] - min_prefix[target1])
			if target2 in min_prefix:
				candidate = max(candidate, P[j + 1] - min_prefix[target2])
			
			if candidate != -10**18:
				ans = max(ans, candidate)
			
			if nums[j] in min_prefix:
				if P[j] < min_prefix[nums[j]]:
					min_prefix[nums[j]] = P[j]
			else:
				min_prefix[nums[j]] = P[j]
		
		return ans if ans != -10**18 else 0