import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n_contests = int(data[0])
	contests = []
	index = 1
	for i in range(n_contests):
		l = int(data[index]); r = int(data[index+1]); index += 2
		contests.append((l, r))
	
	q = int(data[index]); index += 1
	queries = []
	for i in range(q):
		x = int(data[index]); index += 1
		queries.append(x)
	
	max_x = 500000
	size = 1
	while size < max_x:
		size *= 2
	
	n = max_x
	tree_min = [0] * (2 * size)
	tree_max = [0] * (2 * size)
	lazy = [0] * (2 * size)
	
	INF = 10**18
	for i in range(max_x):
		tree_min[size + i] = i + 1
		tree_max[size + i] = i + 1
	for i in range(max_x, size):
		tree_min[size + i] = INF
		tree_max[size + i] = -INF
		
	for i in range(size - 1, 0, -1):
		tree_min[i] = min(tree_min[2*i], tree_min[2*i+1])
		tree_max[i] = max(tree_max[2*i], tree_max[2*i+1])
	
	def apply(node, delta):
		tree_min[node] += delta
		tree_max[node] += delta
		lazy[node] += delta
		
	def push(node):
		if lazy[node] != 0:
			apply(2*node, lazy[node])
			apply(2*node+1, lazy[node])
			lazy[node] = 0
			
	def update(l, r, delta, node=1, segl=0, segr=size-1):
		if r < segl or l > segr:
			return
		if l <= segl and segr <= r:
			apply(node, delta)
			return
		push(node)
		mid = (segl + segr) // 2
		update(l, r, delta, 2*node, segl, mid)
		update(l, r, delta, 2*node+1, mid+1, segr)
		tree_min[node] = min(tree_min[2*node], tree_min[2*node+1])
		tree_max[node] = max(tree_max[2*node], tree_max[2*node+1])
		
	def query_first_ge(l_bound, node=1, segl=0, segr=size-1):
		if tree_min[node] >= l_bound:
			return segl + 1
		if segl == segr:
			return max_x + 1
		push(node)
		mid = (segl + segr) // 2
		if tree_min[2*node] >= l_bound:
			return query_first_ge(l_bound, 2*node, segl, mid)
		else:
			return query_first_ge(l_bound, 2*node+1, mid+1, segr)
			
	def query_last_le(u_bound, node=1, segl=0, segr=size-1):
		if tree_max[node] <= u_bound:
			return segr + 1
		if segl == segr:
			if tree_min[node] <= u_bound:
				return segl + 1
			else:
				return -1
		push(node)
		mid = (segl + segr) // 2
		if tree_min[2*node+1] > u_bound:
			return query_last_le(u_bound, 2*node, segl, mid)
		else:
			if tree_max[2*node+1] <= u_bound:
				return segr + 1
			else:
				return query_last_le(u_bound, 2*node+1, mid+1, segr)
				
	def point_query(index, node=1, segl=0, segr=size-1):
		if segl == segr:
			return tree_min[node]
		push(node)
		mid = (segl + segr) // 2
		if index <= mid:
			return point_query(index, 2*node, segl, mid)
		else:
			return point_query(index, 2*node+1, mid+1, segr)
			
	for (l, r) in contests:
		A = query_first_ge(l)
		if A > max_x:
			continue
		B = query_last_le(r)
		if B == -1:
			continue
		if A <= B:
			update(A-1, B-1, 1)
			
	for x in queries:
		res = point_query(x-1)
		print(res)

if __name__ == "__main__":
	main()