import sys
sys.setrecursionlimit(1 << 25)

class SegmentTree:
	def __init__(self, size):
		self.n = size
		self.min_tree = [10**9] * (4 * size)
		self.max_tree = [-10**9] * (4 * size)
		self.build(1, 1, size)
	
	def build(self, node, l, r):
		if l == r:
			self.min_tree[node] = l
			self.max_tree[node] = l
			return
		mid = (l + r) // 2
		self.build(node * 2, l, mid)
		self.build(node * 2 + 1, mid + 1, r)
		self.min_tree[node] = min(self.min_tree[node * 2], self.min_tree[node * 2 + 1])
		self.max_tree[node] = max(self.max_tree[node * 2], self.max_tree[node * 2 + 1])
	
	def update(self, node, l, r, index, present):
		if l == r:
			if present:
				self.min_tree[node] = l
				self.max_tree[node] = l
			else:
				self.min_tree[node] = 10**9
				self.max_tree[node] = -10**9
			return
		mid = (l + r) // 2
		if index <= mid:
			self.update(node * 2, l, mid, index, present)
		else:
			self.update(node * 2 + 1, mid + 1, r, index, present)
		self.min_tree[node] = min(self.min_tree[node * 2], self.min_tree[node * 2 + 1])
		self.max_tree[node] = max(self.max_tree[node * 2], self.max_tree[node * 2 + 1])
	
	def query_min(self, node, l, r, ql, qr):
		if qr < l or ql > r:
			return 10**9
		if ql <= l and r <= qr:
			return self.min_tree[node]
		mid = (l + r) // 2
		left_min = self.query_min(node * 2, l, mid, ql, qr)
		right_min = self.query_min(node * 2 + 1, mid + 1, r, ql, qr)
		return min(left_min, right_min)
	
	def query_max(self, node, l, r, ql, qr):
		if qr < l or ql > r:
			return -10**9
		if ql <= l and r <= qr:
			return self.max_tree[node]
		mid = (l + r) // 2
		left_max = self.query_max(node * 2, l, mid, ql, qr)
		right_max = self.query_max(node * 2 + 1, mid + 1, r, ql, qr)
		return max(left_max, right_max)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	H = int(data[0])
	W = int(data[1])
	Q = int(data[2])
	queries = []
	index = 3
	for i in range(Q):
		r = int(data[index])
		c = int(data[index + 1])
		index += 2
		queries.append((r, c))
	
	alive = [[True] * (W + 1) for _ in range(H + 1)]
	
	row_trees = [None] * (H + 1)
	for i in range(1, H + 1):
		row_trees[i] = SegmentTree(W)
	
	col_trees = [None] * (W + 1)
	for j in range(1, W + 1):
		col_trees[j] = SegmentTree(H)
	
	removed_count = 0
	for (r, c) in queries:
		if alive[r][c]:
			alive[r][c] = False
			row_trees[r].update(1, 1, W, c, False)
			col_trees[c].update(1, 1, H, r, False)
			removed_count += 1
		else:
			walls_to_remove = []
			if c > 1:
				left_col = row_trees[r].query_max(1, 1, W, 1, c - 1)
				if left_col != -10**9:
					walls_to_remove.append((r, left_col))
			if c < W:
				right_col = row_trees[r].query_min(1, 1, W, c + 1, W)
				if right_col != 10**9:
					walls_to_remove.append((r, right_col))
			if r > 1:
				up_row = col_trees[c].query_max(1, 1, H, 1, r - 1)
				if up_row != -10**9:
					walls_to_remove.append((up_row, c))
			if r < H:
				down_row = col_trees[c].query_min(1, 1, H, r + 1, H)
				if down_row != 10**9:
					walls_to_remove.append((down_row, c))
			
			for (i, j) in walls_to_remove:
				if alive[i][j]:
					alive[i][j] = False
					row_trees[i].update(1, 1, W, j, False)
					col_trees[j].update(1, 1, H, i, False)
					removed_count += 1
					
	total_walls = H * W
	print(total_walls - removed_count)

if __name__ == "__main__":
	main()