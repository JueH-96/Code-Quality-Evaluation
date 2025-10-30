mod = 998244353

def main():
	import sys
	n, m = map(int, sys.stdin.readline().split())
	term1 = pow(m - 1, n, mod)
	if n % 2 == 0:
		ans = (term1 + (m - 1)) % mod
	else:
		ans = (term1 - (m - 1)) % mod
	print(ans)

if __name__ == "__main__":
	main()