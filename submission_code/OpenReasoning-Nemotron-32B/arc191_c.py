import math

def sieve(n):
	is_prime = [True] * (n + 1)
	is_prime[0] = False
	is_prime[1] = False
	for i in range(2, int(math.isqrt(n)) + 1):
		if is_prime[i]:
			for j in range(i * i, n + 1, i):
				is_prime[j] = False
	primes = [i for i in range(2, n + 1) if is_prime[i]]
	return primes

precomputed_primes = sieve(31622)
small_primes = [p for p in precomputed_primes if p <= 1000]

def is_prime(n, small_primes_list):
	if n < 2:
		return False
	if n == 2:
		return True
	if n % 2 == 0:
		return False
	for p in small_primes_list:
		if p * p > n:
			break
		if n % p == 0:
			return False
	d = n - 1
	s = 0
	while d % 2 == 0:
		d //= 2
		s += 1
	bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
	for a in bases:
		if a >= n:
			continue
		x = pow(a, d, n)
		if x == 1 or x == n - 1:
			continue
		composite = True
		for _ in range(s - 1):
			x = (x * x) % n
			if x == n - 1:
				composite = False
				break
		if composite:
			return False
	return True

def factorize(n, primes_list):
	factors = set()
	temp = n
	for p in primes_list:
		if p * p > temp:
			break
		if temp % p == 0:
			factors.add(p)
			while temp % p == 0:
				temp //= p
	if temp > 1:
		factors.add(temp)
	return factors

def main():
	import sys
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index])
		index += 1
		if N == 1:
			results.append("2 1")
		elif N <= 60:
			A = 2
			M = (1 << N) - 1
			results.append(f"{A} {M}")
		else:
			factors_N = factorize(N, precomputed_primes)
			if N % 2 == 0:
				k_list = range(1, 2001)
			else:
				k_list = range(1, 2001, 2)
			found_solution = False
			for k in k_list:
				P = k * N + 1
				if P > 10**18:
					continue
				if not is_prime(P, small_primes):
					continue
				factors_k = factorize(k, precomputed_primes)
				factors_P_minus_1 = factors_N.union(factors_k)
				g = None
				for candidate_g in range(2, 12):
					if candidate_g >= P:
						break
					flag = True
					for q in factors_P_minus_1:
						exponent = (P - 1) // q
						if pow(candidate_g, exponent, P) == 1:
							flag = False
							break
					if flag:
						g = candidate_g
						break
				if g is None:
					continue
				A0 = pow(g, (P - 1) // N, P)
				if N % 2 == 0:
					M_val = 4 * P
					if A0 % 2 == 0:
						A_val = A0 + P
					else:
						A_val = A0
				else:
					M_val = 2 * P
					if A0 % 2 == 0:
						A_val = A0 + P
					else:
						A_val = A0
				results.append(f"{A_val} {M_val}")
				found_solution = True
				break
			if not found_solution:
				results.append("2 1")
	for res in results:
		print(res)

if __name__ == '__main__':
	main()