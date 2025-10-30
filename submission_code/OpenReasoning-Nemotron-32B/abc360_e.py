mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	N = int(data[0])
	K = int(data[1])
	
	if N == 1:
		print(1)
		return
		
	invN = pow(N, mod-2, mod)
	T = (N - 2) * invN % mod
	Tk = pow(T, K, mod)
	part2 = (N - 1) * Tk % mod
	numerator = (N + 1 - part2) % mod
	inv2 = pow(2, mod-2, mod)
	ans = numerator * inv2 % mod
	print(ans)

if __name__ == '__main__':
	main()