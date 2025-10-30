mod = 998244353

def main():
	n = int(input().strip())
	d = len(str(n))
	base = pow(10, d, mod)
	numerator = (pow(base, n, mod) - 1) % mod
	denominator = (base - 1) % mod

	if denominator == 0:
		result = (n % mod) * (n % mod) % mod
	else:
		inv_denom = pow(denominator, mod-2, mod)
		geometric_sum = numerator * inv_denom % mod
		result = (n % mod) * geometric_sum % mod

	print(result)

if __name__ == '__main__':
	main()