mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	suff = [0] * (n + 1)
	for i in range(n - 1, -1, -1):
		suff[i] = (A[i] + suff[i + 1]) % mod

	inv_n = pow(n, mod - 2, mod)
	
	T = 0
	ans = 0
	for i in range(n - 1, -1, -1):
		ans = (suff[i] + T) * inv_n % mod
		T = (T + ans) % mod
		
	print(ans)

if __name__ == '__main__':
	main()