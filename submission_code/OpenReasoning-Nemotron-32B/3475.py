class Solution:
	def minOperations(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 3:
			target = (1 - nums[0]) % 2
			if target == (1 - nums[1]) % 2 and target == (1 - nums[2]) % 2:
				return target
			else:
				return -1
		
		x = [0] * (n - 2)
		x[0] = (1 - nums[0]) % 2
		x[1] = (1 - nums[1] - x[0]) % 2
		
		for j in range(2, n - 2):
			x[j] = (1 - nums[j] - x[j-2] - x[j-1]) % 2
		
		if x[n-3] % 2 != (1 - nums[n-1]) % 2:
			return -1
		
		if (x[n-4] + x[n-3]) % 2 != (1 - nums[n-2]) % 2:
			return -1
		
		return sum(x)