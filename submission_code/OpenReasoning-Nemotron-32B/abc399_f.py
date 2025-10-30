mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	N = int(data[0])
	K = int(data[1])
	A = list(map(int, data[2:2+N]))
	
	P = [0] * (N+1)
	for i in range(1, N+1):
		P[i] = (P[i-1] + A[i-1]) % mod

	binomK = [1]
	for j in range(1, K+1):
		next_val = binomK[-1] * (K - j + 1) // j
		binomK.append(next_val)
	
	power_sums = [0] * (K+1)
	power_sums[0] = 1
	
	ans = 0
	for j in range(1, N+1):
		x = P[j]
		powers = [1] * (K+1)
		for exponent in range(1, K+1):
			powers[exponent] = powers[exponent-1] * x % mod
		
		for t in range(0, K+1):
			term = binomK[t] * powers[t] % mod
			term = term * power_sums[K-t] % mod
			if (K - t) % 2 == 1:
				ans = (ans - term) % mod
			else:
				ans = (ans + term) % mod
		
		for exponent in range(0, K+1):
			power_sums[exponent] = (power_sums[exponent] + powers[exponent]) % mod
	
	ans %= mod
	if ans < 0:
		ans += mod
	print(ans)

if __name__ == '__main__':
	main()