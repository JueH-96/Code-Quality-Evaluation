MOD = 998244353
MAX = 3000000

fact = [0] * (MAX + 1)
inv = [0] * (MAX + 1)
inv_fact = [0] * (MAX + 1)

def precompute():
	fact[0] = 1
	for i in range(1, MAX + 1):
		fact[i] = fact[i-1] * i % MOD
	
	inv[1] = 1
	for i in range(2, MAX + 1):
		inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD
	
	inv_fact[0] = 1
	for i in range(1, MAX + 1):
		inv_fact[i] = inv_fact[i-1] * inv[i] % MOD

def nCr(n, r):
	if r < 0 or r > n:
		return 0
	return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD

def f(w, h):
	if w < 0 or h < 0:
		return 0
	return (nCr(w + h + 2, w + 1) - 1) % MOD

def g(w, h, w1, h1):
	if w < 0 or h < 0 or w1 < 0 or h1 < 0:
		return 0
	res = f(w, h)
	res = (res - f(w1 - 1, h)) % MOD
	res = (res - f(w, h1 - 1)) % MOD
	res = (res + f(w1 - 1, h1 - 1)) % MOD
	return res

def main():
	import sys
	data = sys.stdin.read().split()
	W = int(data[0])
	H = int(data[1])
	L = int(data[2])
	R = int(data[3])
	D = int(data[4])
	U = int(data[5])
	
	precompute()
	
	ans = g(W, H, L, D)
	ans = (ans + g(W, H, R + 1, U + 1)) % MOD
	ans = (ans + g(W - R - 1, H, L, D)) % MOD
	ans = (ans + g(W, H - U - 1, L, U + 1)) % MOD
	ans = (ans - g(W - R - 1, H - U - 1, L, D)) % MOD
	
	ans = (ans % MOD + MOD) % MOD
	print(ans)

if __name__ == '__main__':
	main()