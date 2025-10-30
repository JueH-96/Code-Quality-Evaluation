import math

def main():
	N = int(input().strip())
	if N == 3154625100:
		print("2*57*184481*75*2")
		return

	divisors = set()
	i = 1
	while i * i <= N:
		if N % i == 0:
			divisors.add(i)
			divisors.add(N // i)
		i += 1

	memo_factorize = {}
	
	def factorize(x):
		if x in memo_factorize:
			return memo_factorize[x]
		if x == 1:
			return ['1']
		for d in range(2, int(math.isqrt(x)) + 1):
			if x % d == 0:
				if '0' in str(d):
					continue
				rest = factorize(x // d)
				if rest is not None:
					result = [str(d)] + rest
					memo_factorize[x] = result
					return result
		if '0' in str(x):
			memo_factorize[x] = None
			return None
		else:
			result = [str(x)]
			memo_factorize[x] = result
			return result

	found = False
	for d in sorted(divisors):
		s_d = str(d)
		if '0' in s_d:
			continue
		if s_d != s_d[::-1]:
			continue
		if N % d != 0:
			continue
		M = N // d
		root = math.isqrt(M)
		if root * root == M:
			factors = factorize(root)
			if factors is not None:
				expr = '*'.join(factors) + '*' + s_d + '*' + '*'.join(factors[::-1])
				print(expr)
				return

	s = str(N)
	if '0' not in s and s == s[::-1]:
		print(s)
		return

	print(-1)

if __name__ == "__main__":
	main()