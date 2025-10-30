mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	k = int(data[2])
	A = list(map(int, data[3:3+n]))
	
	size = 1 << 20
	dp = [[0] * size for _ in range(m)]
	dp[0][0] = 1
	
	for a in A:
		new_dp = [[0] * size for _ in range(m)]
		for r in range(m):
			for j in range(size):
				new_dp[r][j] = (new_dp[r][j] + dp[r][j]) % mod
		for r in range(m):
			new_r = (r + 1) % m
			for j in range(size):
				if dp[r][j] != 0:
					new_j = j ^ a
					new_dp[new_r][new_j] = (new_dp[new_r][new_j] + dp[r][j]) % mod
		dp = new_dp

	dp[0][0] = (dp[0][0] - 1) % mod
	
	powers = [0] * size
	for j in range(size):
		powers[j] = pow(j, k, mod)
	
	ans = 0
	for j in range(size):
		ans = (ans + dp[0][j] * powers[j]) % mod
	
	print(ans)

if __name__ == "__main__":
	main()