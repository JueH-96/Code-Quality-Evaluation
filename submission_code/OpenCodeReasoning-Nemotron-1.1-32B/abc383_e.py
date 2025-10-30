import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	k = int(next(it))
	edges = []
	for _ in range(m):
		u = int(next(it))
		v = int(next(it))
		w = int(next(it))
		edges.append((u, v, w))
	
	A_list = [int(next(it)) for _ in range(k)]
	B_list = [int(next(it)) for _ in range(k)]
	
	edges_sorted = sorted(edges, key=lambda x: x[2])
	
	parent_mst = list(range(n+1))
	size_mst = [1] * (n+1)
	
	def find_mst(x):
		if parent_mst[x] != x:
			parent_mst[x] = find_mst(parent_mst[x])
		return parent_mst[x]
	
	mst_edges = []
	for u, v, w in edges_sorted:
		ru = find_mst(u)
		rv = find_mst(v)
		if ru != rv:
			mst_edges.append((u, v, w))
			if size_mst[ru] < size_mst[rv]:
				ru, rv = rv, ru
			parent_mst[rv] = ru
			size_mst[ru] += size_mst[rv]
	
	countA = [0] * (n+1)
	for a in A_list:
		countA[a] += 1
	countB = [0] * (n+1)
	for b in B_list:
		countB[b] += 1
		
	parent_match = list(range(n+1))
	size_match = [1] * (n+1)
	a_comp = [0] * (n+1)
	b_comp = [0] * (n+1)
	for i in range(1, n+1):
		a_comp[i] = countA[i]
		b_comp[i] = countB[i]
		
	def find_match(x):
		if parent_match[x] != x:
			parent_match[x] = find_match(parent_match[x])
		return parent_match[x]
	
	total_cost = 0
	mst_edges_sorted = sorted(mst_edges, key=lambda x: x[2])
	for u, v, w in mst_edges_sorted:
		u_root = find_match(u)
		v_root = find_match(v)
		if u_root == v_root:
			continue
		cross1 = min(a_comp[u_root], b_comp[v_root])
		cross2 = min(a_comp[v_root], b_comp[u_root])
		total_cost += w * (cross1 + cross2)
		if size_match[u_root] < size_match[v_root]:
			u_root, v_root = v_root, u_root
		parent_match[v_root] = u_root
		a_comp[u_root] = a_comp[u_root] + a_comp[v_root] - cross1 - cross2
		b_comp[u_root] = b_comp[u_root] + b_comp[v_root] - cross1 - cross2
		size_match[u_root] += size_match[v_root]
		
	print(total_cost)

if __name__ == "__main__":
	main()