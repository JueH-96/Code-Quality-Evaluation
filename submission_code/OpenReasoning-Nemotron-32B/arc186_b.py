MOD = 998244353

import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	children = [[] for _ in range(n+1)]
	for i in range(1, n+1):
		parent = A[i-1]
		children[parent].append(i)
	
	max_n = n
	fact = [1] * (max_n + 1)
	inv_fact = [1] * (max_n + 1)
	
	for i in range(1, max_n + 1):
		fact[i] = fact[i-1] * i % MOD
		
	inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
	for i in range(max_n, 0, -1):
		inv_fact[i-1] = inv_fact[i] * i % MOD

	size = [0] * (n + 1)
	dp = [1] * (n + 1)
	
	order = []
	stack = [0]
	while stack:
		u = stack.pop()
		order.append(u)
		for v in children[u]:
			stack.append(v)
			
	for u in reversed(order):
		if u == 0:
			k = len(children[u])
			total_size = 0
			for v in children[u]:
				total_size += size[v]
			size[u] = total_size
			denom_inv = inv_fact[k]
			prod_dp = 1
			for v in children[u]:
				denom_inv = denom_inv * inv_fact[size[v] - 1] % MOD
				prod_dp = prod_dp * dp[v] % MOD
			dp[u] = fact[total_size] * denom_inv % MOD * prod_dp % MOD
		else:
			k = len(children[u])
			total_size = 1
			for v in children[u]:
				total_size += size[v]
			size[u] = total_size
			denom_inv = inv_fact[k]
			prod_dp = 1
			for v in children[u]:
				denom_inv = denom_inv * inv_fact[size[v] - 1] % MOD
				prod_dp = prod_dp * dp[v] % MOD
			dp[u] = fact[total_size - 1] * denom_inv % MOD * prod_dp % MOD

	print(dp[0] % MOD)

if __name__ == '__main__':
	main()