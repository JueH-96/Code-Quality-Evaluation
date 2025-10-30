class Solution:
	def minimumOperations(self, nums: List[int], target: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		ops = 0
		inc_prev = max(0, target[0] - nums[0])
		dec_prev = max(0, nums[0] - target[0])
		ops = inc_prev + dec_prev
		
		for i in range(1, n):
			inc_curr = max(0, target[i] - nums[i])
			dec_curr = max(0, nums[i] - target[i])
			if inc_curr > inc_prev:
				ops += inc_curr - inc_prev
			if dec_curr > dec_prev:
				ops += dec_curr - dec_prev
			inc_prev = inc_curr
			dec_prev = dec_curr
		
		return ops