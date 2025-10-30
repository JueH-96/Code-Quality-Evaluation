MOD = 10**9 + 7

class Solution:
	def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
		req_arr = [-1] * n
		for end, cnt in requirements:
			req_arr[end] = cnt
		
		dp = [0] * 401
		dp[0] = 1
		
		for i in range(n):
			new_dp = [0] * 401
			for j in range(401):
				if dp[j] == 0:
					continue
				for k in range(i + 1):
					if j + k <= 400:
						new_dp[j + k] = (new_dp[j + k] + dp[j]) % MOD
			dp = new_dp
			if req_arr[i] != -1:
				for j in range(401):
					if j != req_arr[i]:
						dp[j] = 0
		
		return dp[req_arr[n - 1]] % MOD