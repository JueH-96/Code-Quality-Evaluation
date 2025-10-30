class Solution:
	def numberOfSubsequences(self, nums: List[int]) -> int:
		n = len(nums)
		if n < 7:
			return 0
		
		prefix_freq = [[0] * 1001 for _ in range(n)]
		if 1 <= nums[0] <= 1000:
			prefix_freq[0][nums[0]] = 1
		
		for i in range(1, n):
			prefix_freq[i] = prefix_freq[i-1][:]
			num = nums[i]
			if 1 <= num <= 1000:
				prefix_freq[i][num] += 1
		
		ans = 0
		for q in range(2, n-2):
			left_arr = prefix_freq[q-2]
			for r in range(q+2, n-1):
				for s in range(r+2, n):
					product = nums[q] * nums[s]
					if product % nums[r] != 0:
						continue
					required = product // nums[r]
					if 1 <= required <= 1000:
						ans += left_arr[required]
		return ans