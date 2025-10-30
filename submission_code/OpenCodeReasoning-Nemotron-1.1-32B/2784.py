mod = 10**9 + 7

class Solution:
	def sumOfPower(self, nums: List[int]) -> int:
		nums.sort()
		n = len(nums)
		total = 0
		for a in nums:
			total = (total + a * a % mod * a) % mod
		
		if n > 1:
			F = [0] * n
			for i in range(n-2, -1, -1):
				F[i] = (nums[i+1] * nums[i+1] + 2 * F[i+1]) % mod
			for i in range(n-1):
				total = (total + nums[i] * F[i]) % mod
		
		return total