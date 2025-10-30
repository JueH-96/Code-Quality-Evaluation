mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	N = int(data[0])
	M = int(data[1])
	
	base = M - 1
	term1 = pow(base, N, mod)
	term2 = base % mod
	
	if N % 2 == 0:
		ans = (term1 + term2) % mod
	else:
		ans = (term1 - term2) % mod
		
	print(ans)

if __name__ == '__main__':
	main()