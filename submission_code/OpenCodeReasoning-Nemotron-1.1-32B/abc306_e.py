import sys

class Fenw:
	def __init__(self, n):
		self.n = n
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, delta):
		i = index + 1
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def query_prefix(self, index):
		if index < 0:
			return 0
		res = 0
		i = index + 1
		while i:
			res += self.tree[i]
			i -= i & -i
		return res
			
	def query_range(self, l, r):
		if l > r:
			return 0
		res = self.query_prefix(r)
		if l > 0:
			res -= self.query_prefix(l - 1)
		return res

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	k = int(next(it))
	q = int(next(it))
	queries = []
	distinct_vals = set()
	distinct_vals.add(0)
	for _ in range(q):
		x = int(next(it))
		y = int(next(it))
		queries.append((x, y))
		distinct_vals.add(y)
	
	distinct_vals = sorted(distinct_vals)
	comp = {val: idx for idx, val in enumerate(distinct_vals)}
	M = len(distinct_vals)
	
	freq_tree = Fenw(M)
	sum_tree = Fenw(M)
	
	idx0 = comp[0]
	freq_tree.update(idx0, n)
	
	arr = [0] * (n + 1)
	out_lines = []
	
	for x, y in queries:
		old_val = arr[x]
		new_val = y
		idx_old = comp[old_val]
		idx_new = comp[new_val]
		
		freq_tree.update(idx_old, -1)
		freq_tree.update(idx_new, 1)
		sum_tree.update(idx_old, -old_val)
		sum_tree.update(idx_new, new_val)
		arr[x] = new_val
		
		low, high = 0, M - 1
		candidate_index = -1
		while low <= high:
			mid = (low + high) // 2
			count_ge = freq_tree.query_range(mid, M - 1)
			if count_ge >= k:
				candidate_index = mid
				low = mid + 1
			else:
				high = mid - 1
		
		if candidate_index == -1:
			total = 0
		else:
			count_above = freq_tree.query_range(candidate_index + 1, M - 1)
			sum_above = sum_tree.query_range(candidate_index + 1, M - 1)
			total = sum_above + (k - count_above) * distinct_vals[candidate_index]
		
		out_lines.append(str(total))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()