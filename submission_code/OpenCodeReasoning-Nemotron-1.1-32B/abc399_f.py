mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	binom = [0] * (k+1)
	binom[0] = 1
	for t in range(1, k+1):
		binom[t] = binom[t-1] * (k - t + 1) // t
	
	P = [0] * (n+1)
	for i in range(1, n+1):
		P[i] = (P[i-1] + A[i-1]) % mod
	
	S = [0] * (k+1)
	S[0] = 1
	
	ans = 0
	for j in range(1, n+1):
		p_powers = [1] * (k+1)
		if k >= 1:
			p_powers[1] = P[j] % mod
			for d in range(2, k+1):
				p_powers[d] = p_powers[d-1] * P[j] % mod
		
		for t in range(0, k+1):
			term = binom[t] * p_powers[t] % mod
			term = term * S[k-t] % mod
			if (k - t) % 2 == 1:
				ans = (ans - term) % mod
			else:
				ans = (ans + term) % mod
		
		for d in range(0, k+1):
			S[d] = (S[d] + p_powers[d]) % mod
	
	ans %= mod
	if ans < 0:
		ans += mod
	print(ans)

if __name__ == "__main__":
	main()