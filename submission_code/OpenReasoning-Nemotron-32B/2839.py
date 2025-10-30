class Solution:
	def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
		n = len(nums1)
		points = []
		for i in range(n):
			points.append((nums1[i], nums2[i], nums1[i] + nums2[i]))
		
		comp_set = set()
		for a, b, val in points:
			comp_set.add(b)
		for x, y in queries:
			comp_set.add(y)
		comp_list = sorted(comp_set)
		comp_map = {num: idx for idx, num in enumerate(comp_list)}
		size_comp = len(comp_list)
		
		events = []
		for a, b, val in points:
			events.append((a, b, val, 'point'))
		for i, (x, y) in enumerate(queries):
			events.append((x, y, i, 'query'))
		
		events.sort(key=lambda e: (-e[0], 0 if e[3]=='point' else 1))
		
		if size_comp == 0:
			return [-1] * len(queries)
		
		N = 1
		while N < size_comp:
			N *= 2
		tree = [-10**18] * (2 * N)
		
		def update(i, val):
			i0 = i + N
			if tree[i0] < val:
				tree[i0] = val
				i0 //= 2
				while i0:
					tree[i0] = max(tree[2*i0], tree[2*i0+1])
					i0 //= 2
		
		def query(l, r):
			if l > r:
				return -10**18
			l0 = l + N
			r0 = r + N
			res = -10**18
			while l0 <= r0:
				if l0 % 2 == 1:
					res = max(res, tree[l0])
					l0 += 1
				if r0 % 2 == 0:
					res = max(res, tree[r0])
					r0 -= 1
				l0 //= 2
				r0 //= 2
			return res
		
		ans = [-1] * len(queries)
		
		for event in events:
			if event[3] == 'point':
				a_val, b_val, total, _ = event
				pos = comp_map[b_val]
				update(pos, total)
			else:
				x_val, y_val, idx, _ = event
				pos_y = comp_map[y_val]
				res_val = query(pos_y, size_comp-1)
				if res_val == -10**18:
					ans[idx] = -1
				else:
					ans[idx] = res_val
		
		return ans