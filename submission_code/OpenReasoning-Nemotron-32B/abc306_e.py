import sys
import bisect

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	N = int(next(it))
	K = int(next(it))
	Q = int(next(it))
	updates = []
	all_vals = set()
	all_vals.add(0)
	for _ in range(Q):
		x = int(next(it))
		y = int(next(it))
		updates.append((x, y))
		all_vals.add(y)
	
	all_vals = sorted(all_vals)
	m = len(all_vals)
	
	class Fenw:
		def __init__(self, size):
			self.n = size
			self.tree = [0] * (self.n + 1)
		
		def update(self, index, delta):
			i = index
			while i <= self.n:
				self.tree[i] += delta
				i += i & -i
		
		def query(self, index):
			s = 0
			i = index
			while i > 0:
				s += self.tree[i]
				i -= i & -i
			return s
		
		def query_range(self, l, r):
			if l > r:
				return 0
			return self.query(r) - self.query(l - 1)
	
	freq_tree = Fenw(m)
	sum_tree = Fenw(m)
	
	idx0 = bisect.bisect_left(all_vals, 0)
	freq_tree.update(idx0 + 1, N)
	
	A = [0] * N
	out_lines = []
	
	for i in range(Q):
		x, y = updates[i]
		idx_arr = x - 1
		old_val = A[idx_arr]
		A[idx_arr] = y
		
		pos_old = bisect.bisect_left(all_vals, old_val)
		freq_tree.update(pos_old + 1, -1)
		sum_tree.update(pos_old + 1, -old_val)
		
		pos_new = bisect.bisect_left(all_vals, y)
		freq_tree.update(pos_new + 1, 1)
		sum_tree.update(pos_new + 1, y)
		
		low, high = 0, m - 1
		x_threshold = 0
		while low <= high:
			mid = (low + high) // 2
			x_val = all_vals[mid]
			pos_x = mid
			count_ge_x = freq_tree.query_range(pos_x + 1, m)
			
			next_val = x_val + 1
			pos_next = bisect.bisect_left(all_vals, next_val)
			if pos_next < m:
				count_ge_next = freq_tree.query_range(pos_next + 1, m)
			else:
				count_ge_next = 0
			
			if count_ge_x >= K and count_ge_next < K:
				x_threshold = x_val
				break
			elif count_ge_x < K:
				high = mid - 1
			else:
				low = mid + 1
		
		next_val = x_threshold + 1
		pos_next = bisect.bisect_left(all_vals, next_val)
		if pos_next < m:
			sum_gt = sum_tree.query_range(pos_next + 1, m)
			count_gt = freq_tree.query_range(pos_next + 1, m)
		else:
			sum_gt = 0
			count_gt = 0
		
		result = sum_gt + x_threshold * (K - count_gt)
		out_lines.append(str(result))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()