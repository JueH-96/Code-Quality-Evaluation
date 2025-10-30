MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	N = int(data[0])
	M = int(data[1])
	
	primes = [2, 3, 5, 7, 11, 13]
	g = [0] * 64
	
	for a in range(1, M + 1):
		exp_vec = []
		for p in primes:
			cnt = 0
			temp = a
			while temp % p == 0:
				cnt += 1
				temp //= p
			exp_vec.append(cnt)
		
		for bitmask in range(64):
			prod = 1
			for j in range(6):
				if bitmask & (1 << j):
					prod *= exp_vec[j]
			g[bitmask] = (g[bitmask] + prod) % MOD
	
	A = [0] * 64
	for S in range(64):
		total = 0
		for T in range(64):
			if (S | T) == S:
				total = (total + g[T]) % MOD
		A[S] = total
	
	prime_set = [False] * 64
	for a in range(1, M + 1):
		exp_vec = [0] * 6
		for idx, p in enumerate(primes):
			cnt = 0
			temp = a
			while temp % p == 0:
				cnt += 1
				temp //= p
			exp_vec[idx] = cnt
		bit = 0
		for idx in range(6):
			if exp_vec[idx] > 0:
				bit |= (1 << idx)
		prime_set[bit] = True
	
	Q_mask = 0
	for j in range(6):
		for a in range(1, M + 1):
			if a % primes[j] == 0:
				Q_mask |= (1 << j)
				break
	
	total_ans = 0
	for S in range(64):
		if A[S] == 0:
			continue
		if A[S] == 1:
			term = N % MOD
		else:
			term = pow(A[S], N, MOD)
			numerator = A[S] * (term - 1) % MOD
			denom_inv = pow(A[S] - 1, MOD - 2, MOD)
			term = numerator * denom_inv % MOD
		total_ans = (total_ans + term) % MOD
	
	print(total_ans)

if __name__ == '__main__':
	main()