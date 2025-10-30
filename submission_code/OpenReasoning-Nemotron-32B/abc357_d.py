mod = 998244353

def main():
	n = int(input().strip())
	d = len(str(n))
	base = pow(10, d, mod)
	
	if base == 1:
		S = n % mod
	else:
		power_val = pow(base, n, mod)
		numerator = (power_val - 1) % mod
		denominator = (base - 1) % mod
		inv_denom = pow(denominator, mod-2, mod)
		S = numerator * inv_denom % mod
		
	result = (n % mod) * S % mod
	print(result)

if __name__ == '__main__':
	main()