MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	A_arr = [0] * (n + 1)
	for i in range(2, n + 1):
		A_arr[i] = int(next(it))
	
	fact = 1
	for i in range(1, n):
		fact = (fact * i) % MOD
		
	inv = [0] * (n + 1)
	for i in range(1, n + 1):
		inv[i] = pow(i, MOD - 2, MOD)
		
	F1 = [0] * (n + 1)
	F2 = [0] * (n + 1)
	for i in range(2, n + 1):
		term = (2 * inv[i] - 2 * inv[i] * inv[i]) % MOD
		if term < 0:
			term += MOD
		F1[i] = A_arr[i] * fact % MOD * term % MOD
		F2[i] = A_arr[i] * fact % MOD * inv[i] % MOD
		
	prefixF1 = [0] * (n + 1)
	prefixF2 = [0] * (n + 1)
	for i in range(1, n + 1):
		prefixF1[i] = (prefixF1[i - 1] + F1[i]) % MOD
		prefixF2[i] = (prefixF2[i - 1] + F2[i]) % MOD
		
	out_lines = []
	for _ in range(q):
		u = int(next(it))
		v = int(next(it))
		if u > v:
			u, v = v, u
		part1 = prefixF1[u]
		part2 = 0
		if v - 1 >= u + 1:
			part2 = (prefixF2[v - 1] - prefixF2[u]) % MOD
		part2 = (part2 + A_arr[v] * fact) % MOD
		ans = (part1 + part2) % MOD
		out_lines.append(str(ans))
		
	print("
".join(out_lines))

if __name__ == '__main__':
	main()