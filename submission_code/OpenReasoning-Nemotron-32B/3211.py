class Solution:
	class Node:
		__slots__ = ['best', 'left', 'right']
		def __init__(self):
			self.best = None
			self.left = None
			self.right = None

	class SegmentTree:
		def __init__(self, L, R):
			self.L = L
			self.R = R
			self.root = Solution.Node()
		
		def update(self, idx, state):
			self._update(self.root, self.L, self.R, idx, state)
		
		def _update(self, node, l, r, idx, state):
			if l == r:
				if node.best is None:
					node.best = state
				else:
					if state[0] > node.best[0] or (state[0] == node.best[0] and state[1] > node.best[1]):
						node.best = state
				return
			
			mid = (l + r) // 2
			if idx <= mid:
				if node.left is None:
					node.left = Solution.Node()
				self._update(node.left, l, mid, idx, state)
			else:
				if node.right is None:
					node.right = Solution.Node()
				self._update(node.right, mid+1, r, idx, state)
			
			node.best = None
			if node.left and node.left.best is not None:
				node.best = node.left.best
			if node.right and node.right.best is not None:
				if node.best is None:
					node.best = node.right.best
				else:
					if node.right.best[0] > node.best[0] or (node.right.best[0] == node.best[0] and node.right.best[1] > node.best[1]):
						node.best = node.right.best
		
		def query(self, lq, rq):
			return self._query(self.root, self.L, self.R, lq, rq)
		
		def _query(self, node, l, r, lq, rq):
			if node is None or node.best is None or r < lq or l > rq:
				return None
			if lq <= l and r <= rq:
				return node.best
			
			mid = (l + r) // 2
			left_ans = None
			right_ans = None
			if node.left is not None and lq <= mid:
				left_ans = self._query(node.left, l, mid, lq, min(rq, mid))
			if node.right is not None and rq > mid:
				right_ans = self._query(node.right, mid+1, r, max(lq, mid+1), rq)
			
			if left_ans is None:
				return right_ans
			if right_ans is None:
				return left_ans
			if left_ans[0] > right_ans[0] or (left_ans[0] == right_ans[0] and left_ans[1] > right_ans[1]):
				return left_ans
			else:
				return right_ans

	def findMaximumLength(self, nums: List[int]) -> int:
		n = len(nums)
		P = [0] * (n+1)
		for i in range(1, n+1):
			P[i] = P[i-1] + nums[i-1]
		
		L, R = 0, 20000000000
		tree = self.SegmentTree(L, R)
		tree.update(0, (0, 0))
		
		for i in range(n):
			X = P[i+1]
			state = tree.query(0, X)
			new_dp = state[0] + 1
			new_f = 2 * X - state[1]
			tree.update(new_f, (new_dp, X))
		
		return new_dp