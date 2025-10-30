MOD = 998244353

def ntt(a, inverse=False):
	n = len(a)
	j = 0
	for i in range(1, n):
		bit = n >> 1
		while j >= bit:
			j -= bit
			bit >>= 1
		j += bit
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
				v = a[j+step//2] * w % MOD
				a[j] = (u+v) % MOD
				a[j+step//2] = (u-v) % MOD
				w = w * wn % MOD
		step <<= 1

	if inverse:
		inv_n = pow(n, MOD-2, MOD)
		a[:] = [x * inv_n % MOD for x in a]
	return a

def convolution(a, b):
	len_a = len(a)
	len_b = len(b)
	n = 1
	while n < len_a + len_b:
		n <<= 1
	a += [0] * (n - len_a)
	b += [0] * (n - len_b)
	ntt(a)
	ntt(b)
	c = [a[i] * b[i] % MOD for i in range(n)]
	ntt(c, inverse=True)
	del c[len_a+len_b-1:]
	return c

def poly_exp(poly, exponent, mod):
	result = [1]
	base = poly
	while exponent:
		if exponent & 1:
			result = convolution(result, base)
			if len(result) > mod+1:
				result = result[:mod+1]
		base = convolution(base, base)
		if len(base) > mod+1:
			base = base[:mod+1]
		exponent //= 2
	return result

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	N = int(data[0].strip())
	s = data[1].strip()
	k = s.count('1')
	
	poly_non_S = [1, 1, 1]
	m1 = N - k
	if m1 == 0:
		P1 = [1]
	else:
		P1 = poly_exp(poly_non_S, m1, 2*N)
	
	poly_S = [1, 1, 2]
	P2 = poly_exp(poly_S, k, 2*N)
	
	P = convolution(P1, P2)
	if len(P) < 2*N+1:
		P += [0] * (2*N+1 - len(P))
	total = 0
	for E in range(0, 2*N+1):
		L = max(0, E - N)
		R = min(k, E)
		if L <= R:
			count = R - L + 1
		else:
			count = 0
		total = (total + P[E] * count) % MOD
	print(total)

if __name__ == '__main__':
	main()