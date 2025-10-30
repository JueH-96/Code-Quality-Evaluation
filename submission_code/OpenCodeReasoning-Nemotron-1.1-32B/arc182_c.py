MOD = 998244353

def mat_mult(A, B, mod):
	n = len(A)
	m = len(B)
	p = len(B[0])
	C = [[0] * p for _ in range(n)]
	for i in range(n):
		for k in range(m):
			if A[i][k]:
				for j in range(p):
					C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
	return C

def mat_pow(matrix, power, mod):
	n = len(matrix)
	result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
	base = matrix
	while power:
		if power & 1:
			result = mat_mult(result, base, mod)
		base = mat_mult(base, base, mod)
		power //= 2
	return result

def main():
	import sys
	data = sys.stdin.read().split()
	N = int(data[0])
	M_val = int(data[1])
	
	primes = [2, 3, 5, 7, 11, 13]
	exp_vectors = {}
	for a in range(1, M_val + 1):
		vec = []
		for p in primes:
			cnt = 0
			temp = a
			while temp % p == 0:
				cnt += 1
				temp //= p
			vec.append(cnt)
		exp_vectors[a] = vec

	A_dict = {}
	for bitmask in range(1, 1 << 6):
		S = set()
		for i in range(6):
			if bitmask & (1 << i):
				S.add(primes[i])
		total = 0
		for a in range(1, M_val + 1):
			prod = 1
			for p in S:
				idx = primes.index(p)
				exp_val = exp_vectors[a][idx]
				prod = (prod * exp_val) % MOD
			total = (total + prod) % MOD
		A_dict[frozenset(S)] = total

	F_t = [0] * 7
	if M_val == 1:
		for t in range(0, 7):
			if t + 1 > N + 1:
				F_t[t] = 0
			else:
				num = 1
				for i in range(0, t + 1):
					num = (num * ((N + 1 - i) % MOD)) % MOD
				den = pow(t + 1, -1, MOD)
				F_t[t] = num * den % MOD
	else:
		d = 7
		A_mat = [[0] * d for _ in range(d)]
		for i in range(d):
			if i == 0:
				A_mat[i][0] = M_val % MOD
			else:
				A_mat[i][i] = M_val % MOD
				A_mat[i][i - 1] = (i * M_val) % MOD

		M_mat = [[0] * (d + 1) for _ in range(d + 1)]
		for i in range(d):
			for j in range(d):
				M_mat[i][j] = A_mat[i][j]
		for i in range(d):
			M_mat[i][d] = 1 if i == 0 else 0
		for i in range(d + 1):
			M_mat[d][i] = 0
		M_mat[d][d] = 1

		U0 = [0] * (d + 1)
		U0[0] = 1
		U0[d] = 1

		M_exp = mat_pow(M_mat, N, MOD)
		UN = [0] * (d + 1)
		for i in range(d + 1):
			for j in range(d + 1):
				UN[i] = (UN[i] + M_exp[i][j] * U0[j]) % MOD
		for t in range(d):
			F_t[t] = UN[t]

	inv_M_t = [1] * 7
	if M_val % MOD != 0:
		inv_M = pow(M_val, -1, MOD)
		for t in range(1, 7):
			inv_M_t[t] = inv_M_t[t - 1] * inv_M % MOD

	total_ans = 0

	def partitions(set_):
		if not set_:
			yield []
			return
		first = next(iter(set_))
		for smaller in partitions(set_ - {first}):
			for n, subset in enumerate(smaller):
				yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
			yield [[first]] + smaller

	P_set = set(primes)
	for bitmask in range(0, 1 << 6):
		S = set()
		for i in range(6):
			if bitmask & (1 << i):
				S.add(primes[i])
		if not S:
			term = (F_t[0] - 1) % MOD
			total_ans = (total_ans + term) % MOD
		else:
			for partition in partitions(S):
				t = len(partition)
				c = 1
				for part in partition:
					key = frozenset(part)
					c = (c * A_dict[key]) % MOD
				if t == 0:
					term = c * (F_t[0] - 1) % MOD
				else:
					term = c * inv_M_t[t] % MOD * F_t[t] % MOD
				total_ans = (total_ans + term) % MOD

	total_ans %= MOD
	if total_ans < 0:
		total_ans += MOD
	print(total_ans)

if __name__ == '__main__':
	main()