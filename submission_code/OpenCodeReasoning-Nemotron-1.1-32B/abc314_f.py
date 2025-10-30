import sys

mod = 998244353

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	matches = []
	index = 1
	for i in range(n-1):
		p = int(data[index])
		q = int(data[index+1])
		index += 2
		matches.append((p-1, q-1))
	
	total_nodes = 2 * n
	parent_dsu = list(range(n))
	root_node = list(range(n))
	size_dsu = [1] * n
	children = [[] for _ in range(total_nodes)]
	size_tree = [1] * total_nodes
	
	def find(x):
		if parent_dsu[x] != x:
			parent_dsu[x] = find(parent_dsu[x])
		return parent_dsu[x]
	
	new_id = n
	for p, q in matches:
		rep1 = find(p)
		rep2 = find(q)
		node_id = new_id
		new_id += 1
		left_child = root_node[rep1]
		right_child = root_node[rep2]
		children[node_id] = [left_child, right_child]
		size_tree[node_id] = size_dsu[rep1] + size_dsu[rep2]
		
		if size_dsu[rep1] < size_dsu[rep2]:
			parent_dsu[rep1] = rep2
			size_dsu[rep2] += size_dsu[rep1]
			root_node[rep2] = node_id
		else:
			parent_dsu[rep2] = rep1
			size_dsu[rep1] += size_dsu[rep2]
			root_node[rep1] = node_id
			
	root_id = new_id - 1
	ans = [0] * n
	stack = [(root_id, 0)]
	while stack:
		u, acc = stack.pop()
		if u < n:
			ans[u] = acc
		else:
			left = children[u][0]
			right = children[u][1]
			s = size_tree[u]
			inv_s = pow(s, mod-2, mod)
			left_acc = (acc + size_tree[left] * inv_s) % mod
			right_acc = (acc + size_tree[right] * inv_s) % mod
			stack.append((left, left_acc))
			stack.append((right, right_acc))
			
	print(" ".join(str(x) for x in ans))

if __name__ == "__main__":
	main()