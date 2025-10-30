import sys

class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, i, val):
		while i <= self.n:
			if val > self.tree[i]:
				self.tree[i] = val
			i += i & -i
			
	def query(self, i):
		res = 0
		while i > 0:
			if self.tree[i] > res:
				res = self.tree[i]
			i -= i & -i
		return res

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	q = int(data[1])
	A = list(map(int, data[2:2 + n]))
	queries = []
	index_ptr = 2 + n
	for i in range(q):
		R = int(data[index_ptr])
		X = int(data[index_ptr + 1])
		index_ptr += 2
		queries.append((R, X))
		
	vals = set(A)
	for R, X in queries:
		vals.add(X)
		
	sorted_vals = sorted(vals)
	comp_map = {}
	for idx, val in enumerate(sorted_vals):
		comp_map[val] = idx + 1
		
	M = len(sorted_vals)
	fenw_tree = Fenw(M)
	
	sorted_queries = []
	for idx, (R, X) in enumerate(queries):
		sorted_queries.append((R, X, idx))
	sorted_queries.sort(key=lambda x: x[0])
	
	ans_arr = [0] * q
	curr_index = 0
	for R, X, orig_idx in sorted_queries:
		while curr_index < R:
			a_val = A[curr_index]
			c = comp_map[a_val]
			if c > 1:
				best = fenw_tree.query(c - 1)
			else:
				best = 0
			dp_val = best + 1
			fenw_tree.update(c, dp_val)
			curr_index += 1
			
		c_x = comp_map[X]
		res = fenw_tree.query(c_x)
		ans_arr[orig_idx] = res
		
	for ans in ans_arr:
		print(ans)

if __name__ == "__main__":
	main()