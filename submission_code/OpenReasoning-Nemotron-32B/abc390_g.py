MOD = 998244353

import sys
sys.setrecursionlimit(1 << 20)

def ntt(a, inverse=False):
	n = len(a)
	j = 0
	for i in range(1, n):
		bit = n >> 1
		while j & bit:
			j ^= bit
			bit >>= 1
		j ^= bit
		if i < j:
			a[i], a[j] = a[j], a[i]
	step = 2
	while step <= n:
		wn = pow(3, (MOD-1)//step, MOD)
		if inverse:
			wn = pow(wn, MOD-2, MOD)
		for i in range(0, n, step):
			w = 1
			for j in range(i, i+step//2):
				u = a[j]
				v = w * a[j+step//2] % MOD
				a[j] = (u + v) % MOD
				a[j+step//2] = (u - v) % MOD
				w = w * wn % MOD
		step <<= 1
	if inverse:
		inv_n = pow(n, MOD-2, MOD)
		a[:] = [x * inv_n % MOD for x in a]

def convolution(a, b):
	n1 = len(a)
	n2 = len(b)
	n = 1
	while n < n1 + n2 - 1:
		n <<= 1
	a_ntt = a + [0] * (n - n1)
	b_ntt = b + [0] * (n - n2)
	ntt(a_ntt)
	ntt(b_ntt)
	c = [a_ntt[i] * b_ntt[i] % MOD for i in range(n)]
	ntt(c, inverse=True)
	return c[:n1+n2-1]

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	N = int(data[0])
	if N == 0:
		print(0)
		return

	max_n = N
	fact = [1] * (max_n+1)
	inv_fact = [1] * (max_n+1)
	for i in range(1, max_n+1):
		fact[i] = fact[i-1] * i % MOD
	inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
	for i in range(max_n, 0, -1):
		inv_fact[i-1] = inv_fact[i] * i % MOD

	L_arr = [0] * (N+1)
	total_digits = 0
	for i in range(1, N+1):
		L_i = len(str(i))
		L_arr[i] = L_i
		total_digits += L_i

	a_i_arr = [0] * (N+1)
	for i in range(1, N+1):
		L_i = L_arr[i]
		a_i = pow(10, L_i, MOD)
		a_i_arr[i] = pow(a_i, MOD-2, MOD)

	A = 0
	for i in range(1, N+1):
		A = (A + i * a_i_arr[i]) % MOD

	groups = {}
	for i in range(1, N+1):
		L = L_arr[i]
		if L not in groups:
			groups[L] = [0, 0]
		groups[L][0] += 1
		groups[L][1] = (groups[L][1] + i) % MOD

	c = [0] * (N)
	for L, (cnt, S_val) in groups.items():
		base = pow(10, L, MOD)
		base = pow(base, MOD-2, MOD)
		base_power = pow(base, 2, MOD)
		for k in range(0, N):
			c[k] = (c[k] + S_val * base_power) % MOD
			base_power = base_power * base % MOD

	poly = [1]
	for L in sorted(groups.keys()):
		cnt = groups[L][0]
		base = pow(10, L, MOD)
		base = pow(base, MOD-2, MOD)
		poly_L = [0] * (cnt+1)
		for j in range(0, cnt+1):
			poly_L[j] = fact[cnt] * inv_fact[j] % MOD * inv_fact[cnt - j] % MOD * pow(base, j, MOD) % MOD
		poly = convolution(poly, poly_L)

	C = [0] * (N+1)
	for i, coef in enumerate(poly):
		if i < len(C):
			C[i] = coef

	R = [0] * (N)
	for k in range(0, N):
		if k % 2 == 0:
			R[k] = c[k]
		else:
			R[k] = (-c[k]) % MOD

	conv_full = convolution(poly, R)
	conv = conv_full[:N]

	D_arr = [0] * (N)
	for s in range(0, N):
		if s == 0:
			D_arr[s] = 0
		else:
			if s-1 < len(conv):
				D_arr[s] = conv[s-1]
			else:
				D_arr[s] = 0

	total_sum_inner = 0
	for s in range(0, N):
		term1 = A * C[s] % MOD
		term2 = D_arr[s]
		term = (term1 - term2) % MOD
		denom = inv_fact[N-1-s]
		term = term * denom % MOD
		total_sum_inner = (total_sum_inner + term) % MOD

	total_sum = fact[N-1] * pow(10, total_digits, MOD) % MOD * total_sum_inner % MOD
	total_sum = total_sum % MOD
	if total_sum < 0:
		total_sum += MOD
	print(total_sum)

if __name__ == '__main__':
	main()