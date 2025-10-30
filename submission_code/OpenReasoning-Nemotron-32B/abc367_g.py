mod = 998244353
import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	N = int(next(it))
	M = int(next(it))
	K = int(next(it))
	A = [int(next(it)) for _ in range(N)]
	
	size = 1 << 20
	dp = [[0] * size for _ in range(M)]
	next_dp = [[0] * size for _ in range(M)]
	dp[0][0] = 1

	for a in A:
		for r in range(M):
			for x in range(size):
				next_dp[r][x] = dp[r][x]
				
		for r in range(1, M):
			for x in range(size):
				next_dp[r][x] = (next_dp[r][x] + dp[r-1][x ^ a]) % mod
		for x in range(size):
			next_dp[0][x] = (next_dp[0][x] + dp[M-1][x ^ a]) % mod
		
		dp, next_dp = next_dp, dp

	dp[0][0] = (dp[0][0] - 1) % mod
	if dp[0][0] < 0:
		dp[0][0] += mod

	ans = 0
	for x in range(size):
		cnt = dp[0][x]
		if cnt:
			term = pow(x, K, mod)
			ans = (ans + term * cnt) % mod
	print(ans)

if __name__ == '__main__':
	main()