import sys

class SegmentTree:
	__slots__ = ('size', 'tree', 'lazy', 'inf')
	def __init__(self, H, W):
		self.inf = 10**18
		self.size = 1
		while self.size < W:
			self.size *= 2
		self.tree = [self.inf] * (2 * self.size)
		self.lazy = [self.inf] * (2 * self.size)
		for i in range(W):
			self.tree[self.size + i] = H + 1
		for i in range(self.size - 1, 0, -1):
			self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
	
	def push(self, i):
		if self.lazy[i] == self.inf:
			return
		if i < self.size:
			if self.lazy[2*i] > self.lazy[i]:
				self.lazy[2*i] = self.lazy[i]
				self.tree[2*i] = min(self.tree[2*i], self.lazy[2*i])
			if self.lazy[2*i+1] > self.lazy[i]:
				self.lazy[2*i+1] = self.lazy[i]
				self.tree[2*i+1] = min(self.tree[2*i+1], self.lazy[2*i+1])
		self.lazy[i] = self.inf

	def update(self, l, r, v):
		self._update(l, r, v, 1, 0, self.size-1)
	
	def _update(self, l, r, v, i, segl, segr):
		if r < segl or l > segr:
			return
		if l <= segl and segr <= r:
			if v < self.lazy[i]:
				self.lazy[i] = v
				self.tree[i] = min(self.tree[i], v)
			return
		self.push(i)
		mid = (segl + segr) // 2
		self._update(l, r, v, 2*i, segl, mid)
		self._update(l, r, v, 2*i+1, mid+1, segr)
		self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
	
	def query(self, l, r):
		return self._query(l, r, 1, 0, self.size-1)
	
	def _query(self, l, r, i, segl, segr):
		if r < segl or l > segr:
			return self.inf
		if l <= segl and segr <= r:
			return self.tree[i]
		self.push(i)
		mid = (segl + segr) // 2
		left_res = self._query(l, r, 2*i, segl, mid)
		right_res = self._query(l, r, 2*i+1, mid+1, segr)
		return min(left_res, right_res)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	H = int(next(it)); W = int(next(it)); N = int(next(it))
	bars = []
	for i in range(N):
		r = int(next(it)); c = int(next(it)); L = int(next(it))
		bars.append((r, c, L, i))
	
	bars.sort(key=lambda x: x[0], reverse=True)
	
	seg = SegmentTree(H, W)
	res = [0] * N

	for bar in bars:
		r0, c, L, idx = bar
		l0 = c - 1
		r0_col = c + L - 2
		q = seg.query(l0, r0_col)
		final_row = min(H, q - 1)
		res[idx] = final_row
		seg.update(l0, r0_col, final_row)
	
	for i in range(N):
		print(res[i])

if __name__ == "__main__":
	main()