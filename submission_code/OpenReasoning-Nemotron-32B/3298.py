class Solution:
	def maxSelectedElements(self, nums: List[int]) -> int:
		if not nums:
			return 0
		min_val = min(nums)
		max_val = max(nums) + 1
		count = [0] * (max_val + 2)
		for x in nums:
			if x < len(count):
				count[x] += 1
			if x + 1 < len(count):
				count[x + 1] += 1
		
		residual = 0
		current_streak = 0
		max_streak = 0
		for v in range(min_val, max_val + 1):
			available = count[v] + residual
			if available > 0:
				if residual > 0:
					residual = count[v]
				else:
					residual = count[v] - 1
				current_streak += 1
				if current_streak > max_streak:
					max_streak = current_streak
			else:
				current_streak = 0
				residual = count[v]
		return max_streak