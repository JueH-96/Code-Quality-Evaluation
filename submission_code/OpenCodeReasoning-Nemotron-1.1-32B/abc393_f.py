import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	A = [int(next(it)) for _ in range(n)]
	queries = []
	all_vals = set()
	for i in range(q):
		R = int(next(it))
		X = int(next(it))
		queries.append((R, X, i))
		all_vals.add(X)
	
	for a in A:
		all_vals.add(a)
	
	all_vals = sorted(all_vals)
	comp = {}
	for idx, val in enumerate(all_vals, start=1):
		comp[val] = idx
	m = len(all_vals)
	
	size = 4 * m
	tree = [0] * size
	
	def update(pos, value, node=1, l=1, r=m):
		if l == r:
			if value > tree[node]:
				tree[node] = value
			return
		mid = (l + r) // 2
		if pos <= mid:
			update(pos, value, node * 2, l, mid)
		else:
			update(pos, value, node * 2 + 1, mid + 1, r)
		tree[node] = max(tree[node * 2], tree[node * 2 + 1])
	
	def query(q_l, q_r, node=1, l=1, r=m):
		if q_l > q_r:
			return 0
		if q_l <= l and r <= q_r:
			return tree[node]
		mid = (l + r) // 2
		left_val = 0
		if q_l <= mid:
			left_val = query(q_l, q_r, node * 2, l, mid)
		right_val = 0
		if q_r > mid:
			right_val = query(q_l, q_r, node * 2 + 1, mid + 1, r)
		return max(left_val, right_val)
	
	queries.sort(key=lambda x: x[0])
	ans = [0] * q
	ptr = 0
	
	for i in range(n):
		a_val = A[i]
		c_val = comp[a_val]
		if c_val > 1:
			best = query(1, c_val - 1)
		else:
			best = 0
		dp_i = best + 1
		update(c_val, dp_i)
		
		while ptr < len(queries) and queries[ptr][0] == i + 1:
			R, X, idx = queries[ptr]
			c_x = comp[X]
			res = query(1, c_x)
			ans[idx] = res
			ptr += 1
			
	for a in ans:
		print(a)

if __name__ == '__main__':
	main()