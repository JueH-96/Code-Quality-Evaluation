mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	dp = [0] * (1 << 11)
	dp[1] = 1
	
	for a in A:
		new_dp = [0] * (1 << 11)
		count_big = a - min(a, 10)
		for mask in range(1 << 11):
			if dp[mask] == 0:
				continue
			new_dp[mask] = (new_dp[mask] + dp[mask] * count_big) % mod
			for x in range(1, min(a, 10) + 1):
				new_mask = mask | (mask << x)
				new_mask &= (1 << 11) - 1
				new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % mod
		dp = new_dp

	total_outcomes = 1
	for a in A:
		total_outcomes = (total_outcomes * a) % mod

	good_outcomes = 0
	for mask in range(1 << 11):
		if mask & (1 << 10):
			good_outcomes = (good_outcomes + dp[mask]) % mod

	inv_total = pow(total_outcomes, mod - 2, mod)
	ans = good_outcomes * inv_total % mod
	print(ans)

if __name__ == '__main__':
	main()