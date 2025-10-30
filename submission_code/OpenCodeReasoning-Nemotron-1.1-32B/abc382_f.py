import sys

INF = 10**10

class SegmentTree:
	def __init__(self, n):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [INF] * (2 * self.size)
		self.lazy = [INF] * (2 * self.size)
	
	def push(self, node):
		if self.lazy[node] != INF:
			left_child = 2 * node
			right_child = 2 * node + 1
			if left_child < len(self.data):
				if self.data[left_child] > self.lazy[node]:
					self.data[left_child] = self.lazy[node]
				if self.lazy[left_child] > self.lazy[node]:
					self.lazy[left_child] = self.lazy[node]
			if right_child < len(self.data):
				if self.data[right_child] > self.lazy[node]:
					self.data[right_child] = self.lazy[node]
				if self.lazy[right_child] > self.lazy[node]:
					self.lazy[right_child] = self.lazy[node]
			self.lazy[node] = INF

	def update_range(self, l, r, v, node, segl, segr):
		if r < segl or l > segr:
			return
		if l <= segl and segr <= r:
			if v < self.data[node]:
				self.data[node] = v
			if v < self.lazy[node]:
				self.lazy[node] = v
			return
		self.push(node)
		mid = (segl + segr) // 2
		self.update_range(l, r, v, 2*node, segl, mid)
		self.update_range(l, r, v, 2*node+1, mid+1, segr)
		self.data[node] = min(self.data[2*node], self.data[2*node+1])
	
	def query_range(self, l, r, node, segl, segr):
		if r < segl or l > segr:
			return INF
		if l <= segl and segr <= r:
			return self.data[node]
		self.push(node)
		mid = (segl + segr) // 2
		left_val = self.query_range(l, r, 2*node, segl, mid)
		right_val = self.query_range(l, r, 2*node+1, mid+1, segr)
		return min(left_val, right_val)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	H = int(data[0])
	W = int(data[1])
	N = int(data[2])
	bars = []
	index = 3
	for i in range(N):
		r = int(data[index])
		c = int(data[index+1])
		l = int(data[index+2])
		index += 3
		bars.append((r, c, l, i))
	
	bars_sorted = sorted(bars, key=lambda x: x[0], reverse=True)
	
	seg_tree = SegmentTree(W)
	ans = [0] * N
	
	for (r, c, l, idx) in bars_sorted:
		low_col = c - 1
		high_col = c + l - 2
		res = seg_tree.query_range(low_col, high_col, 1, 0, seg_tree.size-1)
		if res >= INF:
			ans[idx] = H
		else:
			ans[idx] = res - 1
		seg_tree.update_range(low_col, high_col, ans[idx], 1, 0, seg_tree.size-1)
	
	for a in ans:
		print(a)

if __name__ == "__main__":
	main()