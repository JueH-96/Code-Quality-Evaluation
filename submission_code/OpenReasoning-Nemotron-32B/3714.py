mod = 10**9 + 7

class Solution:
	def minMaxSums(self, nums: List[int], k: int) -> int:
		n = len(nums)
		nums.sort()
		r0 = k - 1
		
		T_arr = [0] * (n + 1)
		for x in range(0, n + 1):
			end = min(r0, x)
			total_val = 0
			term = 1
			for j in range(0, end + 1):
				if j > 0:
					term = term * (x - j + 1) % mod
					term = term * pow(j, mod - 2, mod) % mod
				total_val = (total_val + term) % mod
			T_arr[x] = total_val

		total_min = 0
		for i in range(n):
			R = n - i - 1
			count_min = T_arr[R]
			total_min = (total_min + nums[i] * count_min) % mod

		total_max = 0
		for i in range(n):
			L = i
			count_max = T_arr[L]
			total_max = (total_max + nums[i] * count_max) % mod

		total = (total_min + total_max) % mod
		return total