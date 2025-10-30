import sys

class SegmentTree:
	def __init__(self, data, func, identity):
		self.n = len(data)
		self.func = func
		self.identity = identity
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.tree = [identity] * (2 * self.size)
		for i in range(self.n):
			self.tree[self.size + i] = data[i]
		for i in range(self.size - 1, 0, -1):
			self.tree[i] = self.func(self.tree[2*i], self.tree[2*i+1])
	
	def query(self, l, r):
		l += self.size
		r += self.size
		res = self.identity
		while l <= r:
			if l % 2 == 1:
				res = self.func(res, self.tree[l])
				l += 1
			if r % 2 == 0:
				res = self.func(res, self.tree[r])
				r -= 1
			l //= 2
			r //= 2
		return res

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	q = int(next(it))
	
	journeys = []
	for _ in range(m):
		s = int(next(it))
		t = int(next(it))
		if s < t:
			a = s
			b = t - 1
		else:
			a = s - 1
			b = t
		journeys.append((a, b))
	
	left_arr = [-1] * m
	last_occ = [-1] * (n + 1)
	for i in range(m):
		a_i, b_i = journeys[i]
		left_arr[i] = last_occ[a_i]
		last_occ[b_i] = i
		
	right_arr = [10**9] * m
	last_occ = [10**9] * (n + 1)
	for i in range(m - 1, -1, -1):
		a_i, b_i = journeys[i]
		right_arr[i] = last_occ[a_i]
		last_occ[b_i] = i
		
	seg_max = SegmentTree(left_arr, max, -10**9)
	seg_min = SegmentTree(right_arr, min, 10**9)
	
	out_lines = []
	for _ in range(q):
		L = int(next(it))
		R = int(next(it))
		l0 = L - 1
		r0 = R - 1
		max_left = seg_max.query(l0, r0)
		min_right = seg_min.query(l0, r0)
		if max_left >= l0 or min_right <= r0:
			out_lines.append("No")
		else:
			out_lines.append("Yes")
			
	print("
".join(out_lines))

if __name__ == '__main__':
	main()