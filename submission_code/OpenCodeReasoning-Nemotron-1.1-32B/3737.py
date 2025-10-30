class Solution:
	def minCost(self, n: int, cost: List[List[int]]) -> int:
		m = n // 2
		INF = 10**15
		dp_prev = [[INF] * 3 for _ in range(3)]
		for a in range(3):
			for b in range(3):
				if a != b:
					dp_prev[a][b] = cost[0][a] + cost[n-1][b]
		
		if m == 1:
			ans = INF
			for a in range(3):
				for b in range(3):
					if dp_prev[a][b] < ans:
						ans = dp_prev[a][b]
			return ans
		
		for i in range(1, m):
			min_val_excluding_b = [[INF] * 3 for _ in range(3)]
			for a_prev in range(3):
				for b in range(3):
					best = INF
					for b_prev in range(3):
						if b_prev == b:
							continue
						if dp_prev[a_prev][b_prev] < best:
							best = dp_prev[a_prev][b_prev]
					min_val_excluding_b[a_prev][b] = best
			
			dp_curr = [[INF] * 3 for _ in range(3)]
			for a in range(3):
				for b in range(3):
					if a == b:
						continue
					best_prev = INF
					for a_prev in range(3):
						if a_prev == a:
							continue
						if min_val_excluding_b[a_prev][b] < best_prev:
							best_prev = min_val_excluding_b[a_prev][b]
					total = best_prev + cost[i][a] + cost[n-1-i][b]
					dp_curr[a][b] = total
			dp_prev = dp_curr
		
		ans = INF
		for a in range(3):
			for b in range(3):
				if dp_prev[a][b] < ans:
					ans = dp_prev[a][b]
		return ans