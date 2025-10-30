mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	x = int(data[1])
	T = list(map(int, data[2:2+n]))
	
	t0 = T[0]
	s_min = max(0, x - t0 + 1)
	s_max = x
	
	if s_min > s_max:
		print(0)
		return
		
	dp = [0] * (x + 1)
	dp[0] = 1
	inv_n = pow(n, mod - 2, mod)
	
	for s in range(0, x + 1):
		for t_val in T:
			ns = s + t_val
			if ns <= x:
				dp[ns] = (dp[ns] + dp[s] * inv_n) % mod
				
	total = 0
	for s in range(s_min, x + 1):
		total = (total + dp[s]) % mod
		
	ans = total * inv_n % mod
	print(ans)

if __name__ == '__main__':
	main()