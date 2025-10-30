mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	K = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	prefix = [0] * (n+1)
	for i in range(1, n+1):
		prefix[i] = prefix[i-1] + A[i-1]
		
	dp = [0] * (n+1)
	dp[0] = 1
	total_dp = 1
	d = {}
	d[0] = 1
	
	for i in range(1, n+1):
		need = prefix[i] - K
		subtract = d.get(need, 0)
		dp[i] = (total_dp - subtract) % mod
		
		total_dp = (total_dp + dp[i]) % mod
		
		current_prefix = prefix[i]
		if current_prefix in d:
			d[current_prefix] = (d[current_prefix] + dp[i]) % mod
		else:
			d[current_prefix] = dp[i]
			
	print(dp[n] % mod)

if __name__ == '__main__':
	main()