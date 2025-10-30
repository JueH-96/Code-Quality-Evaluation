class Solution:
	def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
		dp = [[10**18] * (op2 + 1) for _ in range(op1 + 1)]
		dp[0][0] = 0
		
		for x in nums:
			choices = []
			choices.append((0, 0, x))
			
			choices.append((1, 0, (x + 1) // 2))
			
			if x >= k:
				choices.append((0, 1, x - k))
				
			if x >= k:
				cand = (x - k + 1) // 2
				if (x + 1) // 2 >= k:
					cand = min(cand, (x + 1) // 2 - k)
				choices.append((1, 1, cand))
				
			new_dp = [[10**18] * (op2 + 1) for _ in range(op1 + 1)]
			for a in range(op1 + 1):
				for b in range(op2 + 1):
					if dp[a][b] == 10**18:
						continue
					for c1, c2, val in choices:
						na = a + c1
						nb = b + c2
						if na <= op1 and nb <= op2:
							if new_dp[na][nb] > dp[a][b] + val:
								new_dp[na][nb] = dp[a][b] + val
			dp = new_dp
		
		ans = 10**18
		for a in range(op1 + 1):
			for b in range(op2 + 1):
				if dp[a][b] < ans:
					ans = dp[a][b]
		return ans