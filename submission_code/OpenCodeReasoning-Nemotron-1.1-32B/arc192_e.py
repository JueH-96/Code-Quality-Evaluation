MOD = 998244353
MAX = 2000000 + 10

def precompute_factorials(max_n, mod=MOD):
	fact = [1] * (max_n + 1)
	inv_fact = [1] * (max_n + 1)
	inv = [1] * (max_n + 1)
	for i in range(2, max_n + 1):
		inv[i] = inv[mod % i] * (mod - mod // i) % mod
	for i in range(1, max_n + 1):
		fact[i] = fact[i - 1] * i % mod
	inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
	for i in range(max_n, 0, -1):
		inv_fact[i - 1] = inv_fact[i] * i % mod
	return fact, inv_fact

fact_global, inv_fact_global = precompute_factorials(MAX, MOD)

def nCr(n, r):
	if r < 0 or r > n:
		return 0
	return fact_global[n] * inv_fact_global[r] % MOD * inv_fact_global[n - r] % MOD

def F(a, b):
	if a < 0 or b < 0:
		return 0
	n = a + b + 4
	k = a + 2
	comb = nCr(n, k)
	return (comb - a - b - 4) % MOD

def main():
	import sys
	data = sys.stdin.read().split()
	W = int(data[0])
	H = int(data[1])
	L = int(data[2])
	R = int(data[3])
	D = int(data[4])
	U = int(data[5])
	
	A = (W + 1) * (H + 1) - (R - L + 1) * (U - D + 1)
	S1 = F(W, H)
	S2 = F(R, U)
	S3 = F(L - 1, U)
	S4 = F(R, D - 1)
	S5 = F(L - 1, D - 1)
	
	total = (S1 - S2 + S3 + S4 - S5 - A) % MOD
	if total < 0:
		total += MOD
	print(total)

if __name__ == '__main__':
	main()