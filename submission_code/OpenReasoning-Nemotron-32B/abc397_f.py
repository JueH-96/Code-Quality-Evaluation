import sys

class SegmentTree:
	def __init__(self, data):
		self.n = len(data)
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.tree = [0] * (2 * self.size)
		self.lazy = [0] * (2 * self.size)
		for i in range(self.n):
			self.tree[self.size + i] = data[i]
		for i in range(self.size - 1, 0, -1):
			self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
	
	def apply(self, idx, val):
		self.tree[idx] += val
		if idx < self.size:
			self.lazy[idx] += val
			
	def push(self, idx):
		if self.lazy[idx] != 0:
			left_child = 2 * idx
			right_child = 2 * idx + 1
			self.tree[left_child] += self.lazy[idx]
			self.tree[right_child] += self.lazy[idx]
			if left_child < self.size:
				self.lazy[left_child] += self.lazy[idx]
			if right_child < self.size:
				self.lazy[right_child] += self.lazy[idx]
			self.lazy[idx] = 0

	def update_range(self, l, r, val, idx=1, segL=0, segR=None):
		if segR is None:
			segR = self.size - 1
		if r < segL or l > segR:
			return
		if l <= segL and segR <= r:
			self.apply(idx, val)
			return
		self.push(idx)
		mid = (segL + segR) // 2
		self.update_range(l, r, val, 2*idx, segL, mid)
		self.update_range(l, r, val, 2*idx+1, mid+1, segR)
		self.tree[idx] = max(self.tree[2*idx], self.tree[2*idx+1])
		
	def query(self, l, r, idx=1, segL=0, segR=None):
		if segR is None:
			segR = self.size - 1
		if r < segL or l > segR:
			return -10**18
		if l <= segL and segR <= r:
			return self.tree[idx]
		self.push(idx)
		mid = (segL + segR) // 2
		left_val = self.query(l, r, 2*idx, segL, mid)
		right_val = self.query(l, r, 2*idx+1, mid+1, segR)
		return max(left_val, right_val)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	prefix = [0] * (n+1)
	freq = [0] * (n+1)
	distinct = 0
	for i in range(n):
		x = A[i]
		if freq[x] == 0:
			distinct += 1
		freq[x] += 1
		prefix[i+1] = distinct
		
	suffix = [0] * (n+1)
	freq = [0] * (n+1)
	distinct = 0
	for i in range(n-1, -1, -1):
		x = A[i]
		if freq[x] == 0:
			distinct += 1
		freq[x] += 1
		suffix[i] = distinct
		
	next_occ = [n] * n
	last_occurrence = [n] * (n+1)
	for i in range(n-1, -1, -1):
		x = A[i]
		if last_occurrence[x] < n:
			next_occ[i] = last_occurrence[x]
		last_occurrence[x] = i
		
	T = [0] * n
	for j in range(n):
		T[j] = prefix[j] + suffix[j]
		
	seg_tree = SegmentTree(T)
	
	best = 0
	for i in range(0, n-1):
		max_val = seg_tree.query(i+1, n-1)
		candidate = prefix[i] + max_val
		if candidate > best:
			best = candidate
			
		l_update = i+1
		r_update = next_occ[i]
		if r_update > n-1:
			r_update = n-1
		if l_update <= r_update:
			seg_tree.update_range(l_update, r_update, -1)
			
	print(best)

if __name__ == "__main__":
	main()