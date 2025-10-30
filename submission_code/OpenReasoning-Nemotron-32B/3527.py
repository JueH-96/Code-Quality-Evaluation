import bisect

class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, idx, delta):
		i = idx + 1
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def query(self, idx):
		if idx < 0:
			return 0
		i = idx + 1
		s = 0
		while i:
			s += self.tree[i]
			i -= i & -i
		return s
		
	def range_query(self, l, r):
		if l > r:
			return 0
		return self.query(r) - self.query(l - 1)

def gap_len(a, b, n):
	if a == b:
		return n - 1
	if a < b:
		return b - a - 1
	else:
		return n - a - 1 + b

class Solution:
	def numberOfAlternatingGroups(self, colors, queries):
		n = len(colors)
		diff = [0] * n
		for i in range(n):
			if colors[i] != colors[(i + 1) % n]:
				diff[i] = 1
			else:
				diff[i] = 0
				
		zeros = sorted([i for i in range(n) if diff[i] == 0])
		k = len(zeros)
		gaps = []
		if k == 0:
			gaps = [n]
		else:
			for i in range(k):
				a = zeros[i]
				b = zeros[(i + 1) % k]
				gaps.append(gap_len(a, b, n))
				
		size_fenw = n + 1
		fenw_count = Fenw(size_fenw)
		fenw_sum = Fenw(size_fenw)
		
		for R in gaps:
			fenw_count.update(R, 1)
			fenw_sum.update(R, R + 1)
			
		ans = []
		for query in queries:
			if query[0] == 1:
				size_i = query[1]
				x = size_i - 1
				if x > n:
					ans.append(0)
				else:
					S1 = fenw_sum.range_query(x, n)
					S2 = fenw_count.range_query(x, n)
					res = S1 - x * S2
					ans.append(res)
			else:
				pos = query[1]
				new_color = query[2]
				old_color = colors[pos]
				if old_color == new_color:
					continue
				colors[pos] = new_color
				indices_to_update = [(pos - 1) % n, pos % n]
				for idx in indices_to_update:
					a = colors[idx]
					b = colors[(idx + 1) % n]
					new_val = 1 if a != b else 0
					old_val = diff[idx]
					if old_val == new_val:
						continue
					diff[idx] = new_val
					if new_val == 0:
						k_curr = len(zeros)
						if k_curr == 0:
							fenw_count.update(n, -1)
							fenw_sum.update(n, -(n + 1))
							fenw_count.update(n - 1, 1)
							fenw_sum.update(n - 1, n)
							bisect.insort(zeros, idx)
						else:
							pos_idx = bisect.bisect_left(zeros, idx)
							if pos_idx == 0:
								left_zero = zeros[-1]
							else:
								left_zero = zeros[pos_idx - 1]
							if pos_idx == k_curr:
								right_zero = zeros[0]
							else:
								right_zero = zeros[pos_idx]
							old_gap = gap_len(left_zero, right_zero, n)
							g1 = gap_len(left_zero, idx, n)
							g2 = gap_len(idx, right_zero, n)
							fenw_count.update(old_gap, -1)
							fenw_sum.update(old_gap, -(old_gap + 1))
							fenw_count.update(g1, 1)
							fenw_sum.update(g1, g1 + 1)
							fenw_count.update(g2, 1)
							fenw_sum.update(g2, g2 + 1)
							bisect.insort(zeros, idx)
					else:
						k_curr = len(zeros)
						pos_idx = bisect.bisect_left(zeros, idx)
						if k_curr == 1:
							fenw_count.update(n - 1, -1)
							fenw_sum.update(n - 1, -n)
							fenw_count.update(n, 1)
							fenw_sum.update(n, n + 1)
							zeros.pop(pos_idx)
						else:
							if pos_idx == 0:
								left_zero = zeros[-1]
							else:
								left_zero = zeros[pos_idx - 1]
							if pos_idx == k_curr - 1:
								right_zero = zeros[0]
							else:
								right_zero = zeros[pos_idx + 1]
							g1 = gap_len(left_zero, idx, n)
							g2 = gap_len(idx, right_zero, n)
							new_gap = gap_len(left_zero, right_zero, n)
							fenw_count.update(g1, -1)
							fenw_sum.update(g1, -(g1 + 1))
							fenw_count.update(g2, -1)
							fenw_sum.update(g2, -(g2 + 1))
							fenw_count.update(new_gap, 1)
							fenw_sum.update(new_gap, new_gap + 1)
							zeros.pop(pos_idx)
		return ans