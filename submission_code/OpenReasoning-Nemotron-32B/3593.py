import math

class Solution:
	def maxScore(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 1:
			return nums[0] * nums[0]
		
		def lcm(a, b):
			return a * b // math.gcd(a, b)
		
		prefix_gcd = [0] * n
		prefix_lcm = [0] * n
		prefix_gcd[0] = nums[0]
		prefix_lcm[0] = nums[0]
		for i in range(1, n):
			prefix_gcd[i] = math.gcd(prefix_gcd[i-1], nums[i])
			prefix_lcm[i] = lcm(prefix_lcm[i-1], nums[i])
		
		suffix_gcd = [0] * n
		suffix_lcm = [0] * n
		suffix_gcd[n-1] = nums[n-1]
		suffix_lcm[n-1] = nums[n-1]
		for i in range(n-2, -1, -1):
			suffix_gcd[i] = math.gcd(suffix_gcd[i+1], nums[i])
			suffix_lcm[i] = lcm(suffix_lcm[i+1], nums[i])
		
		ans = prefix_gcd[n-1] * prefix_lcm[n-1]
		
		for i in range(n):
			if i == 0:
				g = suffix_gcd[1]
				l = suffix_lcm[1]
			elif i == n-1:
				g = prefix_gcd[n-2]
				l = prefix_lcm[n-2]
			else:
				g = math.gcd(prefix_gcd[i-1], suffix_gcd[i+1])
				l = lcm(prefix_lcm[i-1], suffix_lcm[i+1])
			score = g * l
			if score > ans:
				ans = score
				
		return ans