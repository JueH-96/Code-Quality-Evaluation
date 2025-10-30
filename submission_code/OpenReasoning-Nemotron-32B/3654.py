class Solution:
	def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
		base_sum = sum(nums)
		n = len(nums)
		dp = [[-10**18] * (op2 + 1) for _ in range(op1 + 1)]
		dp[0][0] = 0
		
		for num in nums:
			options = []
			options.append((0, 0, 0))
			
			red1 = num - (num + 1) // 2
			options.append((red1, 1, 0))
			
			if num >= k:
				options.append((k, 0, 1))
			
			cand1 = -10**18
			cand2 = -10**18
			if num >= k:
				cand1 = num - (num - k + 1) // 2
			if (num + 1) // 2 >= k:
				cand2 = num - ((num + 1) // 2 - k)
			
			red3 = max(cand1, cand2)
			if red3 >= 0:
				options.append((red3, 1, 1))
			
			new_dp = [[-10**18] * (op2 + 1) for _ in range(op1 + 1)]
			for a in range(op1 + 1):
				for b in range(op2 + 1):
					if dp[a][b] == -10**18:
						continue
					for red, c1, c2 in options:
						na = a + c1
						nb = b + c2
						if na <= op1 and nb <= op2:
							if new_dp[na][nb] < dp[a][b] + red:
								new_dp[na][nb] = dp[a][b] + red
			dp = new_dp
		
		max_red = max(max(row) for row in dp)
		return base_sum - max_red