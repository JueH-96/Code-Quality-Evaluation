class Solution:
	def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
		n = len(fruits)
		if n == 0:
			return 0
		
		size = 4 * n
		tree = [0] * size
		
		def build(node, l, r):
			if l == r:
				tree[node] = baskets[l]
				return
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			build(left_child, l, mid)
			build(right_child, mid+1, r)
			tree[node] = max(tree[left_child], tree[right_child])
		
		def update(node, l, r, idx, val):
			if l == r:
				tree[node] = val
				return
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			if idx <= mid:
				update(left_child, l, mid, idx, val)
			else:
				update(right_child, mid+1, r, idx, val)
			tree[node] = max(tree[left_child], tree[right_child])
		
		def query(node, l, r, f):
			if tree[node] < f:
				return -1
			if l == r:
				return l
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			if tree[left_child] >= f:
				return query(left_child, l, mid, f)
			else:
				return query(right_child, mid+1, r, f)
		
		build(0, 0, n-1)
		unplaced = 0
		for fruit in fruits:
			idx = query(0, 0, n-1, fruit)
			if idx == -1:
				unplaced += 1
			else:
				update(0, 0, n-1, idx, -10**18)
				
		return unplaced