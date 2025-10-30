import sys
sys.setrecursionlimit(1 << 25)

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
	
	def push(self, i):
		if self.lazy[i] != 0:
			self.tree[2*i] += self.lazy[i]
			self.tree[2*i+1] += self.lazy[i]
			if 2*i < self.size:
				self.lazy[2*i] += self.lazy[i]
				self.lazy[2*i+1] += self.lazy[i]
			self.lazy[i] = 0

	def update_range(self, l, r, val, i=1, segL=0, segR=None):
		if segR is None:
			segR = self.size - 1
		if r < segL or l > segR:
			return
		if l <= segL and segR <= r:
			self.tree[i] += val
			if i < self.size:
				self.lazy[i] += val
			return
		self.push(i)
		mid = (segL + segR) // 2
		self.update_range(l, r, val, 2*i, segL, mid)
		self.update_range(l, r, val, 2*i+1, mid+1, segR)
		self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
	
	def query_range(self, l, r, i=1, segL=0, segR=None):
		if segR is None:
			segR = self.size - 1
		if r < segL or l > segR:
			return -10**18
		if l <= segL and segR <= r:
			return self.tree[i]
		self.push(i)
		mid = (segL + segR) // 2
		left_res = self.query_range(l, r, 2*i, segL, mid)
		right_res = self.query_range(l, r, 2*i+1, mid+1, segR)
		return max(left_res, right_res)
	
	def update(self, l, r, val):
		self.update_range(l, r, val, 1, 0, self.size-1)
	
	def query(self, l, r):
		return self.query_range(l, r, 1, 0, self.size-1)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	next_occurrence_arr = [n] * n
	next_temp = [n] * (n+1)
	for i in range(n-1, -1, -1):
		x = A[i]
		next_occurrence_arr[i] = next_temp[x]
		next_temp[x] = i

	suf = [0] * (n+2)
	seen_suf = [False] * (n+1)
	count_suf = 0
	for i in range(n-1, -1, -1):
		if not seen_suf[A[i]]:
			count_suf += 1
			seen_suf[A[i]] = True
		suf[i] = count_suf

	pre = [0] * n
	seen_pre = [False] * (n+1)
	count_pre = 0
	for i in range(n):
		if not seen_pre[A[i]]:
			count_pre += 1
			seen_pre[A[i]] = True
		pre[i] = count_pre

	base0 = [0] * n
	for j in range(n):
		if j+1 < n:
			base0[j] = pre[j] + suf[j+1]
		else:
			base0[j] = pre[j]

	if n < 3:
		print(0)
		return

	seg_tree = SegmentTree(base0)
	seen_left = [False] * (n+1)
	left_distinct = 0
	ans = 0

	for i in range(0, n-1):
		x = A[i]
		if not seen_left[x]:
			left_distinct += 1
			seen_left[x] = True

		L = i
		R = next_occurrence_arr[i] - 1
		if R >= n:
			R = n-1
		if L <= R:
			seg_tree.update(L, R, -1)

		if i+1 <= n-2:
			max_val = seg_tree.query(i+1, n-2)
			candidate = left_distinct + max_val
			if candidate > ans:
				ans = candidate

	print(ans)

if __name__ == "__main__":
	main()