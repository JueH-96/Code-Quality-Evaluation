import sys
sys.setrecursionlimit(300000)

class Solution:
	def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
		mod = 10**9 + 7
		n = len(nums)
		size = 4 * n
		tree = [None] * size
		NEG_INF = -10**18
		
		def merge(left, right):
			a1, b1, c1, d1 = left
			a2, b2, c2, d2 = right
			part1 = max(a1, b1, c1, d1)
			part2 = max(a1, c1)
			a = part1 + a2
			b = max(part1 + b2, part2 + d2)
			c = max(c1, d1) + a2
			d = max(c1 + max(b2, d2), d1 + b2)
			return (a, b, c, d)
		
		def build(idx, l, r):
			if l == r:
				tree[idx] = (0, NEG_INF, NEG_INF, nums[l])
				return
			mid = (l + r) // 2
			left_idx = 2 * idx + 1
			right_idx = 2 * idx + 2
			build(left_idx, l, mid)
			build(right_idx, mid + 1, r)
			tree[idx] = merge(tree[left_idx], tree[right_idx])
		
		def update(idx, l, r, pos, x):
			if l == r:
				tree[idx] = (0, NEG_INF, NEG_INF, x)
				return
			mid = (l + r) // 2
			left_idx = 2 * idx + 1
			right_idx = 2 * idx + 2
			if pos <= mid:
				update(left_idx, l, mid, pos, x)
			else:
				update(right_idx, mid + 1, r, pos, x)
			tree[idx] = merge(tree[left_idx], tree[right_idx])
		
		build(0, 0, n - 1)
		total_ans = 0
		for pos, x in queries:
			update(0, 0, n - 1, pos, x)
			root_val = tree[0]
			ans = max(root_val)
			total_ans = (total_ans + ans) % mod
		
		return total_ans