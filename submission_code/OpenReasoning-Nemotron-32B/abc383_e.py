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
	rank_mst = [0] * (n+1)
	
	def find_mst(x):
		root = x
		while root != parent_mst[root]:
			root = parent_mst[root]
		while x != root:
			nxt = parent_mst[x]
			parent_mst[x] = root
			x = nxt
		return root
	
	mst_edges = []
	for u, v, w in edges_sorted:
		if len(mst_edges) == n-1:
			break
		ru = find_mst(u)
		rv = find_mst(v)
		if ru == rv:
			continue
		mst_edges.append((u, v, w))
		if rank_mst[ru] < rank_mst[rv]:
			parent_mst[ru] = rv
		else:
			parent_mst[rv] = ru
			if rank_mst[ru] == rank_mst[rv]:
				rank_mst[ru] += 1
				
	parent_match = list(range(n+1))
	rank_match = [0] * (n+1)
	countA = [0] * (n+1)
	countB = [0] * (n+1)
	
	for a in A_list:
		countA[a] += 1
	for b in B_list:
		countB[b] += 1
		
	def find_match(x):
		root = x
		while root != parent_match[root]:
			root = parent_match[root]
		while x != root:
			nxt = parent_match[x]
			parent_match[x] = root
			x = nxt
		return root
		
	mst_edges_sorted = sorted(mst_edges, key=lambda x: x[2])
	total_cost = 0
	for u, v, w in mst_edges_sorted:
		ru = find_match(u)
		rv = find_match(v)
		if ru == rv:
			continue
		A1 = countA[ru]
		B1 = countB[ru]
		A2 = countA[rv]
		B2 = countB[rv]
		cross_pairs = min(A1, B2) + min(B1, A2)
		total_cost += w * cross_pairs
		
		countA[ru] = A1 - min(A1, B2)
		countB[ru] = B1 - min(B1, A2)
		countA[rv] = A2 - min(B1, A2)
		countB[rv] = B2 - min(A1, B2)
		
		if rank_match[ru] < rank_match[rv]:
			parent_match[ru] = rv
			countA[rv] += countA[ru]
			countB[rv] += countB[ru]
			if rank_match[ru] == rank_match[rv]:
				rank_match[rv] += 1
		else:
			parent_match[rv] = ru
			countA[ru] += countA[rv]
			countB[ru] += countB[rv]
			if rank_match[ru] == rank_match[rv]:
				rank_match[ru] += 1
				
	print(total_cost)

if __name__ == "__main__":
	main()