import sys
sys.setrecursionlimit(300000)

class Node:
	__slots__ = ('max_sub', 'max_prefix', 'max_suffix', 'total')
	def __init__(self, max_sub, max_prefix, max_suffix, total):
		self.max_sub = max_sub
		self.max_prefix = max_prefix
		self.max_suffix = max_suffix
		self.total = total

class SegmentTree:
	def __init__(self, data):
		self.n = len(data)
		self.data = data
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.tree = [None] * (2 * self.size)
		self.build(0, 0, self.n-1)
	
	def build(self, node, l, r):
		if l == r:
			val = self.data[l]
			self.tree[node] = Node(val, val, val, val)
			return
		mid = (l + r) // 2
		left_node = 2 * node + 1
		right_node = 2 * node + 2
		self.build(left_node, l, mid)
		self.build(right_node, mid+1, r)
		self.tree[node] = self.merge_nodes(self.tree[left_node], self.tree[right_node])
	
	def merge_nodes(self, left, right):
		if left is None:
			return right
		if right is None:
			return left
		total = left.total + right.total
		max_prefix = max(left.max_prefix, left.total + right.max_prefix)
		max_suffix = max(right.max_suffix, right.total + left.max_suffix)
		max_sub = max(left.max_sub, right.max_sub, left.max_suffix + right.max_prefix)
		return Node(max_sub, max_prefix, max_suffix, total)
	
	def query_range(self, node, seg_l, seg_r, ql, qr):
		if seg_r < ql or seg_l > qr:
			return None
		if ql <= seg_l and seg_r <= qr:
			return self.tree[node]
		mid = (seg_l + seg_r) // 2
		left_res = None
		right_res = None
		if ql <= mid:
			left_res = self.query_range(2*node+1, seg_l, mid, ql, qr)
		if qr > mid:
			right_res = self.query_range(2*node+2, mid+1, seg_r, ql, qr)
		if left_res is None:
			return right_res
		if right_res is None:
			return left_res
		return self.merge_nodes(left_res, right_res)
	
	def query(self, l, r):
		if l > r:
			return None
		return self.query_range(0, 0, self.n-1, l, r)

class Solution:
	def maxSubarraySum(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		if n == 1:
			return nums[0]
		
		seg_tree = SegmentTree(nums)
		ans = seg_tree.query(0, n-1).max_sub
		
		from collections import defaultdict
		pos_dict = defaultdict(list)
		for i, num in enumerate(nums):
			pos_dict[num].append(i)
		
		for x, positions in pos_dict.items():
			if len(positions) == n:
				continue
			segments = []
			if positions[0] > 0:
				segments.append((0, positions[0]-1))
			for i in range(1, len(positions)):
				if positions[i] - positions[i-1] > 1:
					segments.append((positions[i-1]+1, positions[i]-1))
			if positions[-1] < n-1:
				segments.append((positions[-1]+1, n-1))
			if not segments:
				continue
				
			current_node = None
			for (l, r) in segments:
				seg_node = seg_tree.query(l, r)
				if current_node is None:
					current_node = seg_node
				else:
					current_node = seg_tree.merge_nodes(current_node, seg_node)
			if current_node is not None:
				ans = max(ans, current_node.max_sub)
				
		return ans