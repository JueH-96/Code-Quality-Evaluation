import math

MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n-1]))
	
	dp = {1: 1}
	for a_val in A:
		new_dp = {}
		factorizations = []
		for p in range(1, a_val + 1):
			if a_val % p == 0:
				q = a_val // p
				if math.gcd(p, q) == 1:
					factorizations.append((p, q))
		
		for (p, q) in factorizations:
			for v, ways in dp.items():
				nv = v * q
				new_ways = ways * p * nv % MOD
				new_dp[nv] = (new_dp.get(nv, 0) + new_ways) % MOD
				if p != q:
					nv2 = v * p
					new_ways2 = ways * q * nv2 % MOD
					new_dp[nv2] = (new_dp.get(nv2, 0) + new_ways2) % MOD
		dp = new_dp
	
	ans = 0
	for ways in dp.values():
		ans = (ans + ways) % MOD
	print(ans)

if __name__ == '__main__':
	main()