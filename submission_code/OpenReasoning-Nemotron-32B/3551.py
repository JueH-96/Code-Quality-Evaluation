class Solution:
	def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
		n = len(nums)
		px_all = [0] * (n + 1)
		for i in range(n):
			px_all[i + 1] = px_all[i] ^ nums[i]
		
		B0 = []
		B1 = []
		for i in range(n):
			if i % 2 == 0:
				B0.append(nums[i])
			else:
				B1.append(nums[i])
				
		px0 = [0] * (len(B0) + 1)
		for i in range(len(B0)):
			px0[i + 1] = px0[i] ^ B0[i]
			
		px1 = [0] * (len(B1) + 1)
		for i in range(len(B1)):
			px1[i + 1] = px1[i] ^ B1[i]
			
		M = [[0] * n for _ in range(n)]
		for i in range(n):
			for j in range(i, n):
				L = j - i + 1
				if L & (L - 1) == 0 and L > 0:
					M[i][j] = px_all[j + 1] ^ px_all[i]
				else:
					num_elements = (j - i) // 2 + 1
					if i % 2 == 0:
						start_index = i // 2
						M[i][j] = px0[start_index + num_elements] ^ px0[start_index]
					else:
						start_index = (i - 1) // 2
						M[i][j] = px1[start_index + num_elements] ^ px1[start_index]
		
		dp = [[0] * n for _ in range(n)]
		for i in range(n):
			dp[i][i] = M[i][i]
			
		for length in range(1, n):
			for i in range(n - length):
				j = i + length
				dp[i][j] = max(M[i][j], dp[i + 1][j], dp[i][j - 1])
				
		res = []
		for l, r in queries:
			res.append(dp[l][r])
			
		return res