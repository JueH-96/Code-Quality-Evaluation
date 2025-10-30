import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data:
		return
	it = iter(data)
	N = int(next(it))
	M = int(next(it))
	wheels = []
	for _ in range(N):
		C_i = int(next(it))
		P_i = int(next(it))
		outcomes = [int(next(it)) for _ in range(P_i)]
		wheels.append((C_i, P_i, outcomes))
	
	dp = [0.0] * (M + 1)
	
	for x in range(M - 1, -1, -1):
		best = float('inf')
		for wheel in wheels:
			C_i, P_i, outcomes = wheel
			count_zeros = 0
			total = 0.0
			for s in outcomes:
				if s == 0:
					count_zeros += 1
				else:
					next_state = x + s
					if next_state < M:
						total += dp[next_state]
			denominator = P_i - count_zeros
			if denominator == 0:
				candidate = float('inf')
			else:
				candidate = (C_i * P_i + total) / denominator
			if candidate < best:
				best = candidate
		dp[x] = best
	
	print(dp[0])

if __name__ == '__main__':
	main()