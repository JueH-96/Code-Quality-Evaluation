import math

def main():
	MAX_SKIP = 40
	N = int(input().strip())
	points = []
	for i in range(N):
		data = input().split()
		x = int(data[0])
		y = int(data[1])
		points.append((x, y))
	
	INF = 10**18
	dp = [[INF] * (MAX_SKIP + 1) for _ in range(N)]
	dp[0][0] = 0.0
	
	for j in range(1, N):
		for t in range(0, MAX_SKIP + 1):
			low_bound = max(0, j - t - 1)
			for i in range(low_bound, j):
				k0 = t - (j - i - 1)
				if k0 < 0 or k0 > MAX_SKIP:
					continue
				x1, y1 = points[i]
				x2, y2 = points[j]
				dx = x1 - x2
				dy = y1 - y2
				dist = math.sqrt(dx * dx + dy * dy)
				new_val = dp[i][k0] + dist
				if new_val < dp[j][t]:
					dp[j][t] = new_val
					
	ans = INF
	for t in range(0, MAX_SKIP + 1):
		if dp[N-1][t] >= INF:
			continue
		if t == 0:
			penalty = 0.0
		else:
			penalty = 2**(t - 1)
		total = dp[N-1][t] + penalty
		if total < ans:
			ans = total
			
	print("{:.20f}".format(ans))

if __name__ == '__main__':
	main()