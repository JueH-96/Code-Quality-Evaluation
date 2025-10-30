import random
import math

def is_prime(n):
	if n < 2:
		return False
	if n == 2:
		return True
	if n % 2 == 0:
		return False
	d = n - 1
	s = 0
	while d % 2 == 0:
		s += 1
		d //= 2
	test_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
	for a in test_bases:
		if a >= n:
			continue
		x = pow(a, d, n)
		if x == 1 or x == n - 1:
			continue
		for _ in range(s - 1):
			x = (x * x) % n
			if x == n - 1:
				break
		else:
			return False
	return True

def factorize(x, small_primes):
	factors = set()
	temp = x
	for p in small_primes:
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
	if not data:
		return
	T = int(data[0])
	index = 1
	max_sieve = 35000
	sieve = [True] * (max_sieve + 1)
	sieve[0] = sieve[1] = False
	for i in range(2, int(math.isqrt(max_sieve)) + 1):
		if sieve[i]:
			for j in range(i * i, max_sieve + 1, i):
				sieve[j] = False
	small_primes = [i for i in range(2, max_sieve + 1) if sieve[i]]
	out_lines = []
	for _ in range(T):
		N = int(data[index])
		index += 1
		if N == 1:
			out_lines.append("2 1")
			continue
		candidate = None
		k = 1
		while k <= 1000000:
			candidate_val = k * N + 1
			if candidate_val > 10**18:
				break
			if is_prime(candidate_val):
				candidate = candidate_val
				break
			k += 1
		if candidate is None:
			while True:
				k = random.randint(1, (10**18 - 1) // N)
				candidate_val = k * N + 1
				if candidate_val > 10**18:
					continue
				if is_prime(candidate_val):
					candidate = candidate_val
					break
		P = candidate
		k_val = (P - 1) // N
		factors_N = factorize(N, small_primes)
		factors_k = factorize(k_val, small_primes)
		distinct_primes = factors_N | factors_k
		g = 2
		found_g = False
		while g < P:
			flag = True
			for q in distinct_primes:
				exponent = (P - 1) // q
				if pow(g, exponent, P) == 1:
					flag = False
					break
			if flag:
				found_g = True
				break
			g += 1
		if not found_g:
			g = 2
			while g < P:
				flag = True
				for q in distinct_primes:
					exponent = (P - 1) // q
					if pow(g, exponent, P) == 1:
						flag = False
						break
				if flag:
					found_g = True
					break
				g += 1
		A0 = pow(g, (P - 1) // N, P)
		A = A0
		M = P
		out_lines.append(f"{A} {M}")
	for line in out_lines:
		print(line)

if __name__ == '__main__':
	main()