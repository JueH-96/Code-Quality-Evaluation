MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	dp = [0] * (n + 1)
	dp[0] = 1
	s = [0] * (n + 1)
	s[0] = 1
	j = 0
	for i in range(n):
		while j < n and A[j] <= i:
			j += 1
		if j > i:
			dp[i + 1] = s[i]
		else:
			dp[i + 1] = (s[i] - s[j - 1]) % MOD
		s[i + 1] = (s[i] + dp[i + 1]) % MOD
	print(dp[n] % MOD)

if __name__ == "__main__":
	main()