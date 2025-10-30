BIG = 10**18

import sys

def main():
	data = sys.stdin.read().split()
	if not data: 
		print("No")
		return
	n = int(data[0])
	m = int(data[1])
	dist = [[BIG] * n for _ in range(n)]
	for i in range(n):
		dist[i][i] = 0

	idx = 2
	for i in range(m):
		u = int(data[idx])
		v = int(data[idx+1])
		w = int(data[idx+2])
		idx += 3
		u0 = u - 1
		v0 = v - 1
		if w < dist[u0][v0]:
			dist[u0][v0] = w

	for k in range(n):
		for i in range(n):
			if dist[i][k] == BIG:
				continue
			for j in range(n):
				if dist[k][j] != BIG:
					new_dist = dist[i][k] + dist[k][j]
					if new_dist < dist[i][j]:
						dist[i][j] = new_dist

	dp = [[BIG] * n for _ in range(1 << n)]
	for i in range(n):
		dp[1 << i][i] = 0

	for mask in range(1 << n):
		for i in range(n):
			if dp[mask][i] == BIG:
				continue
			for j in range(n):
				if mask & (1 << j):
					continue
				if dist[i][j] == BIG:
					continue
				new_mask = mask | (1 << j)
				new_cost = dp[mask][i] + dist[i][j]
				if new_cost < dp[new_mask][j]:
					dp[new_mask][j] = new_cost

	ans = min(dp[(1 << n) - 1])
	if ans == BIG:
		print("No")
	else:
		print(ans)

if __name__ == '__main__':
	main()