MOD = 998244353

import sys
sys.setrecursionlimit(1 << 20)

def next_power_of_two(n):
	n -= 1
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	return n + 1

def ntt(a, inverse=False):
	n = len(a)
	j = 0
	for i in range(1, n):
		bit = n >> 1
		while j & bit:
			j ^= bit
			bit >>= 1
		j |= bit
		if i < j:
			a[i], a[j] = a[j], a[i]
	
	length = 2
	while length <= n:
		wlen = pow(3, (MOD-1)//length, MOD)
		if inverse:
			wlen = pow(wlen, MOD-2, MOD)
		for i in range(0, n, length):
			w = 1
			for j in range(i, i+length//2):
				u = a[j]
				v = a[j+length//2] * w % MOD
				a[j] = (u + v) % MOD
				a[j+length//2] = (u - v) % MOD
				w = w * wlen % MOD
		length <<= 1
	if inverse:
		inv_n = pow(n, MOD-2, MOD)
		a[:] = [x * inv_n % MOD for x in a]

def convolution(a, b):
	n1, n2 = len(a), len(b)
	n = next_power_of_two(n1 + n2 - 1)
	a += [0] * (n - n1)
	b += [0] * (n - n2)
	ntt(a)
	ntt(b)
	c = [a[i] * b[i] % MOD for i in range(n)]
	ntt(c, inverse=True)
	return c

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	max_n = 200000
	fac = [1] * (max_n+1)
	for i in range(1, max_n+1):
		fac[i] = fac[i-1] * i % MOD
	ifac = [1] * (max_n+1)
	ifac[max_n] = pow(fac[max_n], MOD-2, MOD)
	for i in range(max_n, 0, -1):
		ifac[i-1] = ifac[i] * i % MOD

	len_count = [0] * 6
	for i in range(1, n+1):
		d = len(str(i))
		if d <= 6:
			len_count[d-1] += 1

	f = [1]
	powers = [10, 100, 1000, 10000, 100000, 1000000]
	for d in range(6):
		count_d = len_count[d]
		c_val = powers[d] % MOD
		poly = [0] * (count_d+1)
		for k in range(count_d+1):
			poly[k] = fac[count_d] * ifac[k] % MOD * ifac[count_d-k] % MOD * pow(c_val, k, MOD) % MOD
		f = convolution(f, poly)
		if len(f) > n+1:
			f = f[:n+1]

	C_arr = [0] * n
	for i in range(n):
		if n-1-i < 0:
			C_arr[i] = 0
		else:
			C_arr[i] = fac[i] * fac[n-1-i] % MOD

	D_arr = [0] * n
	for i in range(n):
		if i < len(f):
			D_arr[i] = f[i]
		else:
			D_arr[i] = 0

	D_rev = list(reversed(D_arr))
	conv_result = convolution(C_arr, D_rev)
	R = [0] * n
	for j in range(n):
		idx = n-1+j
		if idx < len(conv_result):
			R[j] = conv_result[idx] * ifac[n-1] % MOD
		else:
			R[j] = 0

	total_sum = 0
	for d in range(6):
		c_val = powers[d] % MOD
		P_val = 0
		power = 1
		for j in range(n):
			term = R[j] * power % MOD
			if j % 2 == 1:
				term = -term
			P_val = (P_val + term) % MOD
			power = power * c_val % MOD
		low = 10**d
		high = min(n, 10**(d+1)-1)
		if low > n:
			S_d = 0
		else:
			count = high - low + 1
			total_val = low + high
			S_d = (total_val * count) // 2
			S_d %= MOD
		total_sum = (total_sum + S_d * P_val) % MOD

	ans = fac[n-1] * total_sum % MOD
	ans %= MOD
	if ans < 0:
		ans += MOD
	print(ans)

if __name__ == '__main__':
	main()