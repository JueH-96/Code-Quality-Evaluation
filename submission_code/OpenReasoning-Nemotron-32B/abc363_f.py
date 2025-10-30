import math
from collections import defaultdict

def factorize(n):
	factors = {}
	d = 2
	while d * d <= n:
		while n % d == 0:
			factors[d] = factors.get(d, 0) + 1
			n //= d
		d += 1
	if n > 1:
		factors[n] = factors.get(n, 0) + 1
	return factors

def get_all_divisors_from_factors(factors):
	divisors = [1]
	for prime, exp in factors.items():
		new_divisors = []
		for e in range(1, exp + 1):
			power = prime ** e
			for d in divisors:
				new_divisors.append(d * power)
		divisors += new_divisors
	return sorted(set(divisors))

def get_divisors_from_exponents(prime_exponents):
	primes = list(prime_exponents.keys())
	exponents_list = [list(range(prime_exponents[p] + 1)) for p in primes]
	divisors = []
	stack = [(0, 1)]
	while stack:
		i, current = stack.pop()
		if i == len(primes):
			divisors.append(current)
		else:
			for exp in exponents_list[i]:
				stack.append((i + 1, current * (primes[i] ** exp)))
	return divisors

def is_palindrome_without_zeros(x):
	s = str(x)
	if '0' in s:
		return False
	return s == s[::-1]

def main():
	N = int(input().strip())
	if N == 1:
		print("1")
		return

	factors_N = factorize(N)
	divisors_N = get_all_divisors_from_factors(factors_N)
	prime_list = list(factors_N.keys())
	exponents_dict = {p: factors_N[p] for p in prime_list}

	divisors_of_divisors = {}
	for x in divisors_N:
		exp_vector = {}
		temp = x
		for p in prime_list:
			cnt = 0
			while temp % p == 0:
				cnt += 1
				temp //= p
			exp_vector[p] = cnt
		divisors_x = get_divisors_from_exponents(exp_vector)
		divisors_of_divisors[x] = divisors_x

	D_map = {}
	D_set = set()
	max_f = 10**6
	for f in range(1, max_f + 1):
		s = str(f)
		if '0' in s:
			continue
		reverse_f = int(s[::-1])
		d = f * reverse_f
		if d > N:
			continue
		D_map[d] = f
		D_set.add(d)

	divisors_map_D = {}
	for x in divisors_N:
		divisors_x = divisors_of_divisors[x]
		valid_d = []
		for d in divisors_x:
			if d == 1:
				continue
			if d in D_set:
				valid_d.append(d)
		divisors_map_D[x] = valid_d

	best_depth = {}
	found_solution = None
	for depth_limit in range(0, 251):
		res = dfs(N, 0, depth_limit, [], best_depth, divisors_map_D, D_map)
		if res is not None:
			path, final_x = res
			left_factors = path
			middle_factor = final_x
			right_factors = [int(str(f)[::-1]) for f in reversed(path)]
			factors = left_factors + [middle_factor] + right_factors
			s = '*'.join(str(f) for f in factors)
			if len(s) > 1000:
				continue
			found_solution = s
			break

	if found_solution is None:
		print(-1)
	else:
		print(found_solution)

def dfs(x, depth, depth_limit, path, best_depth, divisors_map_D, D_map):
	if depth > depth_limit:
		return None
	if x in best_depth and best_depth[x] <= depth:
		return None
	best_depth[x] = depth

	if is_palindrome_without_zeros(x):
		return (path, x)

	for d in divisors_map_D.get(x, []):
		f_val = D_map[d]
		if x % d != 0:
			continue
		y = x // d
		res = dfs(y, depth + 1, depth_limit, path + [f_val], best_depth, divisors_map_D, D_map)
		if res is not None:
			return res

	return None

if __name__ == "__main__":
	main()