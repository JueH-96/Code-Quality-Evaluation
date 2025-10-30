class Solution:
	def maxValue(self, nums: List[int], k: int) -> int:
		n = len(nums)
		L = [set() for _ in range(n)]
		dp = [[False] * 128 for _ in range(k+1)]
		dp[0][0] = True
		
		for i in range(n):
			new_dp = [row[:] for row in dp]
			for a in range(k):
				for mask in range(128):
					if dp[a][mask]:
						new_mask = mask | nums[i]
						new_dp[a+1][new_mask] = True
			dp = new_dp
			for mask in range(128):
				if dp[k][mask]:
					L[i].add(mask)
		
		R = [set() for _ in range(n)]
		dp = [[False] * 128 for _ in range(k+1)]
		dp[0][0] = True
		
		for i in range(n-1, -1, -1):
			new_dp = [row[:] for row in dp]
			for a in range(k):
				for mask in range(128):
					if dp[a][mask]:
						new_mask = mask | nums[i]
						new_dp[a+1][new_mask] = True
			dp = new_dp
			for mask in range(128):
				if dp[k][mask]:
					R[i].add(mask)
		
		ans = 0
		for i in range(k-1, n - k):
			for mask1 in L[i]:
				for mask2 in R[i+1]:
					candidate = mask1 ^ mask2
					if candidate > ans:
						ans = candidate
		return ans