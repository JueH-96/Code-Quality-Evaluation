import math

def main():
	n = int(input().strip())
	points = []
	for _ in range(n):
		x, y = map(int, input().split())
		points.append((x, y))
	
	if n == 2:
		dx = points[1][0] - points[0][0]
		dy = points[1][1] - points[0][1]
		dist_val = math.sqrt(dx*dx + dy*dy)
		print("{:.20f}".format(dist_val))
		return
	
	M = min(n-2, 50)
	dp = [[10**18] * (M+1) for _ in range(n)]
	dp[0][0] = 0.0
	
	for j in range(n-1):
		end_i = min(n-1, j + M + 1)
		for i in range(j+1, end_i+1):
			skipped = i - j - 1
			dx = points[i][0] - points[j][0]
			dy = points[i][1] - points[j][1]
			dist_val = math.sqrt(dx*dx + dy*dy)
			for c_j in range(M+1):
				if dp[j][c_j] >= 10**18:
					continue
				c_new = c_j + skipped
				if c_new > M:
					continue
				new_val = dp[j][c_j] + dist_val
				if new_val < dp[i][c_new]:
					dp[i][c_new] = new_val
	
	ans = 10**18
	for c in range(M+1):
		if dp[n-1][c] < 10**18:
			penalty = 0 if c == 0 else (2**(c-1))
			total = dp[n-1][c] + penalty
			if total < ans:
				ans = total
	print("{:.20f}".format(ans))

if __name__ == "__main__":
	main()