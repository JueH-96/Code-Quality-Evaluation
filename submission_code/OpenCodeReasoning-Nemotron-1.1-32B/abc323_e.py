mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	X = int(data[1])
	T = list(map(int, data[2:2+n]))
	
	invN = pow(n, mod-2, mod)
	
	dp = [0] * (X + 1)
	dp[0] = 1
	ans = 0

	for s in range(X + 1):
		for j in range(n):
			total_time = s + T[j]
			if total_time > X:
				if j == 0:
					ans = (ans + dp[s] * invN) % mod
			else:
				dp[total_time] = (dp[total_time] + dp[s] * invN) % mod
				
	print(ans % mod)

if __name__ == "__main__":
	main()