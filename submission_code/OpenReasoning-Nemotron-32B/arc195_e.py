MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	q = int(data[1])
	A = [0] * (n + 1)
	idx = 2
	if n >= 2:
		for i in range(2, n + 1):
			A[i] = int(data[idx])
			idx += 1
	
	fact = 1
	for i in range(1, n):
		fact = fact * i % MOD
		
	inv = [0] * (n + 1)
	if n >= 1:
		inv[1] = 1
	for i in range(2, n + 1):
		inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD

	H = [0] * (n + 1)
	prefixH = [0] * (n + 1)
	for i in range(2, n + 1):
		H[i] = A[i] * inv[i] % MOD
		prefixH[i] = (prefixH[i - 1] + H[i]) % MOD

	out_lines = []
	for _ in range(q):
		u = int(data[idx])
		v = int(data[idx + 1])
		idx += 2
		if u > v:
			u, v = v, u
		term_u = 0
		if u >= 2:
			term_u = (A[u] + prefixH[u - 1]) % MOD
		term_v = 0
		if v >= 2:
			term_v = (A[v] + prefixH[v - 1]) % MOD
		term_lca = prefixH[u]
		ans = (term_u + term_v - 2 * term_lca) % MOD
		ans = ans * fact % MOD
		ans = (ans % MOD + MOD) % MOD
		out_lines.append(str(ans))
	
	sys.stdout.write("
".join(out_lines) + "
")

if __name__ == "__main__":
	main()