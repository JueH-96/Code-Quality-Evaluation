class Solution:
	def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
		n = len(nums)
		if all(x == 0 for x in nums):
			return 0
		
		dp = []
		for i in range(n):
			arr = [False] * (nums[i] + 1)
			arr[0] = True
			dp.append(arr)
		
		m = len(queries)
		for k in range(m):
			l, r, val = queries[k]
			for i in range(l, r + 1):
				if dp[i][nums[i]]:
					continue
				for s in range(nums[i], val - 1, -1):
					if dp[i][s - val]:
						dp[i][s] = True
						if s == nums[i]:
							break
			if all(dp[i][nums[i]] for i in range(n)):
				return k + 1
		return -1