class Solution:
	def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
		n = len(fruits)
		INF = -10**18
		dp = [[INF] * n for _ in range(n)]
		dp[0][n-1] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
		
		for k in range(1, n):
			new_dp = [[INF] * n for _ in range(n)]
			for i in range(max(0, n-1-k), n):
				for j in range(max(0, n-1-k), n):
					best = INF
					for i1 in [i, i-1]:
						for j1 in [k - i, k - i - 1]:
							if i1 < 0 or i1 >= n or j1 < 0 or j1 >= n:
								continue
							if i1 + j1 != k - 1:
								continue
							for c in [0, 1, -1]:
								prev_j2 = j + c
								if prev_j2 < 0 or prev_j2 >= n:
									continue
								for j3 in [j, j-1, j+1]:
									if j3 < 0 or j3 >= n:
										continue
									prev_val = dp[i1][j3]
									if prev_val == INF:
										continue
									total = prev_val
									rooms = set()
									rooms.add((i, k - i))
									rooms.add((k, j))
									rooms.add((j, k))
									if (i1, j1) not in rooms:
										total += fruits[i1][j1]
									if (k-1, prev_j2) not in rooms:
										total += fruits[k-1][prev_j2]
									if (j3, k-1) not in rooms and (j3, k-1) != (i1, j1) and (j3, k-1) != (k-1, prev_j2):
										total += fruits[j3][k-1]
									if total > best:
										best = total
					if best != INF:
						new_dp[i][j] = best
			dp = new_dp

		return dp[n-1][n-1]