mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	M_val = int(data[1])
	B = list(map(int, data[2:2+n]))
	
	q_total = sum(1 for x in B if x == -1)
	
	q_arr = [0] * (n+2)
	q_arr[n+1] = 0
	for i in range(n, 0, -1):
		if B[i-1] == -1:
			q_arr[i] = (q_arr[i+1] + 1)
		else:
			q_arr[i] = q_arr[i+1]
			
	dp = [0] * (M_val+1)
	
	if B[0] == -1:
		for x in range(1, M_val+1):
			dp[x] = 1
	else:
		x = B[0]
		dp[x] = 1
		
	total_ways = 0
	
	for i in range(2, n+1):
		ways_i = 0
		if B[i-1] == -1:
			for m_prev in range(1, M_val+1):
				ways_i = (ways_i + dp[m_prev] * (m_prev-1)) % mod
				
			S = [0] * (M_val+2)
			S[M_val] = dp[M_val]
			for k in range(M_val-1, 0, -1):
				S[k] = (S[k+1] + dp[k]) % mod
				
			new_dp = [0] * (M_val+1)
			for k in range(1, M_val+1):
				term1 = S[k+1]
				term2 = dp[k] * (M_val - k + 1) % mod
				new_dp[k] = (term1 + term2) % mod
			dp = new_dp
		else:
			x = B[i-1]
			for m_prev in range(1, M_val+1):
				if x < m_prev:
					ways_i = (ways_i + dp[m_prev]) % mod
					
			new_dp = [0] * (M_val+1)
			for m_prev in range(1, M_val+1):
				k0 = min(m_prev, x)
				new_dp[k0] = (new_dp[k0] + dp[m_prev]) % mod
			dp = new_dp
			
		exp = pow(M_val, q_arr[i+1], mod)
		total_ways = (total_ways + ways_i * exp) % mod
		
	base = pow(M_val, q_total, mod)
	ans = (base + total_ways) % mod
	print(ans)

if __name__ == '__main__':
	main()