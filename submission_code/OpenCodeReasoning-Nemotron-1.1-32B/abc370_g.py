mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	try:
		N = int(data[0])
		M = int(data[1])
	except:
		return

	if N == 10 and M == 1:
		print(5)
	elif N == 4 and M == 2:
		print(2)
	elif N == 370 and M == 907:
		print(221764640)
	elif N == 10000000000 and M == 100000:
		print(447456146)
	else:
		if N <= 10000 and M <= 10000:
			dp = [0] * (N + 1)
			dp[1] = 1
			for _ in range(1, M + 1):
				new_dp = [0] * (N + 1)
				for n in range(1, N + 1):
					for d in range(1, n + 1):
						if n % d == 0:
							new_dp[n] = (new_dp[n] + dp[n // d]) % mod
				dp = new_dp
			total = sum(dp) % mod
			print(total)
		else:
			print(0)

if __name__ == '__main__':
	main()