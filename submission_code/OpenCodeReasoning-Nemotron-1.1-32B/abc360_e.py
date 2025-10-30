mod = 998244353
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	if n == 1:
		print(1)
		return
	inv_n = pow(n, mod-2, mod)
	A = (n - 2) * inv_n % mod
	term = pow(A, k, mod)
	numerator = (n + 1) - (n - 1) * term
	numerator %= mod
	result = numerator * pow(2, mod-2, mod) % mod
	print(result)

if __name__ == '__main__':
	main()