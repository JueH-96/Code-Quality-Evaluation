class Solution:
	def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
		MOD = 10**9 + 7
		n = len(nums)
		size = 4 * n
		tree = [(0, 0)] * size  # (dp0, dp1) for each node

		# Build function
		def build(node, l, r):
			if l == r:
				tree[node] = (0, nums[l])
				return
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			build(left_child, l, mid)
			build(right_child, mid + 1, r)
			l0, l1 = tree[left_child]
			r0, r1 = tree[right_child]
			cand0 = l0 + r0
			cand1 = l0 + r1
			cand2 = l1 + r0
			dp0 = max(cand0, cand1, cand2)
			dp1 = l0 + r1
			tree[node] = (dp0, dp1)
		
		build(0, 0, n - 1)
		
		# Update function
		def update(node, l, r, idx, val):
			if l == r:
				tree[node] = (0, val)
				return
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			if idx <= mid:
				update(left_child, l, mid, idx, val)
			else:
				update(right_child, mid + 1, r, idx, val)
			l0, l1 = tree[left_child]
			r0, r1 = tree[right_child]
			cand0 = l0 + r0
			cand1 = l0 + r1
			cand2 = l1 + r0
			dp0 = max(cand0, cand1, cand2)
			dp1 = l0 + r1
			tree[node] = (dp0, dp1)
		
		total_ans = 0
		for pos, x in queries:
			update(0, 0, n - 1, pos, x)
			root0, root1 = tree[0]
			ans_query = max(root0, root1)
			total_ans = (total_ans + ans_query) % MOD
			
		return total_ans % MOD