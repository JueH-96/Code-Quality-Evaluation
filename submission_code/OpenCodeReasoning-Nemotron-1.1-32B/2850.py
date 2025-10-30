class Solution:
	def longestString(self, x: int, y: int, z: int) -> int:
		NEG = -10**9
		dp = [[[ [NEG, NEG] for _ in range(z+1) ] for _ in range(y+1)] for _ in range(x+1)]
		
		for i in range(x+1):
			for j in range(y+1):
				for k in range(z+1):
					if i >= 1:
						if 2 > dp[i][j][k][0]:
							dp[i][j][k][0] = 2
						if i-1 >= 0 and dp[i-1][j][k][1] != NEG:
							candidate = dp[i-1][j][k][1] + 2
							if candidate > dp[i][j][k][0]:
								dp[i][j][k][0] = candidate
					
					if j >= 1:
						if 2 > dp[i][j][k][1]:
							dp[i][j][k][1] = 2
					if k >= 1:
						if 2 > dp[i][j][k][1]:
							dp[i][j][k][1] = 2
					if j >= 1 and dp[i][j-1][k][0] != NEG:
						candidate = dp[i][j-1][k][0] + 2
						if candidate > dp[i][j][k][1]:
							dp[i][j][k][1] = candidate
					if k >= 1 and dp[i][j][k-1][1] != NEG:
						candidate = dp[i][j][k-1][1] + 2
						if candidate > dp[i][j][k][1]:
							dp[i][j][k][1] = candidate
		
		ans = 0
		for i in range(x+1):
			for j in range(y+1):
				for k in range(z+1):
					for s in range(2):
						if dp[i][j][k][s] > ans:
							ans = dp[i][j][k][s]
		return ans