MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	matches = []
	index = 1
	for i in range(n-1):
		p = int(data[index])
		q = int(data[index+1])
		index += 2
		matches.append((p, q))
	
	max_size = n
	inv_arr = [0] * (max_size + 1)
	if max_size >= 1:
		inv_arr[1] = 1
	for i in range(2, max_size + 1):
		inv_arr[i] = (MOD - MOD // i) * inv_arr[MOD % i] % MOD

	total_nodes = 2 * n
	parent_uf = list(range(total_nodes + 1))
	size_arr = [0] * (total_nodes + 1)
	children = [[] for _ in range(total_nodes + 1)]
	
	for i in range(1, n + 1):
		size_arr[i] = 1
		
	next_node = n + 1

	def find(x):
		stack = []
		while parent_uf[x] != x:
			stack.append(x)
			x = parent_uf[x]
		root = x
		for y in stack:
			parent_uf[y] = root
		return root

	for p, q in matches:
		u = find(p)
		v = find(q)
		w = next_node
		next_node += 1
		
		size_arr[w] = size_arr[u] + size_arr[v]
		children[w] = [u, v]
		
		parent_uf[u] = w
		parent_uf[v] = w
		parent_uf[w] = w

	root = next_node - 1
	stack = [(root, 0)]
	ans = [0] * (n + 1)
	
	while stack:
		node, acc = stack.pop()
		if node <= n:
			ans[node] = acc
		else:
			s = size_arr[node]
			inv_s = inv_arr[s]
			for child in children[node]:
				add = size_arr[child] * inv_s % MOD
				new_acc = (acc + add) % MOD
				stack.append((child, new_acc))
				
	res_list = [str(ans[i]) for i in range(1, n + 1)]
	print(" ".join(res_list))

if __name__ == "__main__":
	main()