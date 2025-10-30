mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	K = int(data[0])
	C_list = list(map(int, data[1:1+26]))
	
	max_n = K
	fact = [1] * (max_n + 1)
	for i in range(1, max_n + 1):
		fact[i] = fact[i - 1] * i % mod
		
	inv_fact = [1] * (max_n + 1)
	inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
	for i in range(max_n, 0, -1):
		inv_fact[i - 1] = inv_fact[i] * i % mod
		
	dp = [0] * (K + 1)
	dp[0] = 1
	
	for c in C_list:
		max_deg = min(c, K)
		poly = [inv_fact[j] for j in range(max_deg + 1)]
		
		new_dp = [0] * (K + 1)
		for i in range(K + 1):
			if dp[i] == 0:
				continue
			end = min(len(poly), K - i + 1)
			for j in range(end):
				new_dp[i + j] = (new_dp[i + j] + dp[i] * poly[j]) % mod
		dp = new_dp
		
	ans = 0
	for n in range(1, K + 1):
		term = fact[n] * dp[n] % mod
		ans = (ans + term) % mod
		
	print(ans)

if __name__ == "__main__":
	main()