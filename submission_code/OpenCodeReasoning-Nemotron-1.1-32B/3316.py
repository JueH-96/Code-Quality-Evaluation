mod = 10**9 + 7

class Solution:
	def sumOfPowers(self, nums: List[int], k: int) -> int:
		n = len(nums)
		arr = sorted(nums)
		
		gap_set = set()
		for i in range(n):
			for j in range(i+1, n):
				gap_set.add(arr[j] - arr[i])
		distinct_gaps = sorted(gap_set)
		m = len(distinct_gaps)
		gap_to_index = {gap: idx for idx, gap in enumerate(distinct_gaps)}
		
		if k == 1:
			return 0
		
		dp = [[0] * m for _ in range(n)]
		
		for i in range(1, n):
			for prev in range(i):
				gap_val = arr[i] - arr[prev]
				g_idx = gap_to_index[gap_val]
				dp[i][g_idx] = (dp[i][g_idx] + 1) % mod
		
		if k == 2:
			total = 0
			for i in range(n):
				for g_idx in range(m):
					total = (total + distinct_gaps[g_idx] * dp[i][g_idx]) % mod
			return total
		
		for j_val in range(3, k+1):
			new_dp = [[0] * m for _ in range(n)]
			for i in range(j_val-1, n):
				for prev in range(i):
					gap_val = arr[i] - arr[prev]
					gap_val_idx = gap_to_index[gap_val]
					for g_idx_prev in range(m):
						count = dp[prev][g_idx_prev]
						if count == 0:
							continue
						g_prev = distinct_gaps[g_idx_prev]
						if g_prev < gap_val:
							new_g_idx = g_idx_prev
						else:
							new_g_idx = gap_val_idx
						new_dp[i][new_g_idx] = (new_dp[i][new_g_idx] + count) % mod
			dp = new_dp
		
		total = 0
		for i in range(n):
			for g_idx in range(m):
				total = (total + distinct_gaps[g_idx] * dp[i][g_idx]) % mod
		return total