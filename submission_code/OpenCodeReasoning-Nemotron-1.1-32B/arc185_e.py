MOD = 998244353
MAX_A = 100000

import sys

def main():
	phi = list(range(MAX_A + 1))
	for i in range(2, MAX_A + 1):
		if phi[i] == i:
			for j in range(i, MAX_A + 1, i):
				phi[j] -= phi[j] // i

	divisors = [[] for _ in range(MAX_A + 1)]
	for i in range(1, MAX_A + 1):
		for j in range(i, MAX_A + 1, i):
			divisors[j].append(i)
	
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1 + n]))
	
	F = [0] * (MAX_A + 1)
	total_dp = 0
	ans = []
	
	for i in range(n):
		a = A[i]
		T_i = 0
		for d in divisors[a]:
			T_i = (T_i + phi[d] * F[d]) % MOD
		
		dp_i = (total_dp + T_i) % MOD
		total_dp = (total_dp + dp_i) % MOD
		ans.append(total_dp)
		
		power = pow(2, i, MOD)
		for d in divisors[a]:
			F[d] = (F[d] + power) % MOD
	
	for res in ans:
		print(res)

if __name__ == "__main__":
	main()