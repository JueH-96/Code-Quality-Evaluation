class Solution:
	def countComponents(self, nums: List[int], threshold: int) -> int:
		n = len(nums)
		S_set = set()
		for num in nums:
			if num <= threshold:
				S_set.add(num)
		
		multiples = [[] for _ in range(threshold+1)]
		for d in range(1, threshold+1):
			for k in range(1, threshold//d + 1):
				x = d * k
				if x in S_set:
					multiples[d].append(x)
		
		parent = list(range(n))
		rank = [0] * n
		
		def find(x):
			if parent[x] != x:
				parent[x] = find(parent[x])
			return parent[x]
		
		def union(x, y):
			rx = find(x)
			ry = find(y)
			if rx == ry:
				return
			if rank[rx] < rank[ry]:
				parent[rx] = ry
			elif rank[rx] > rank[ry]:
				parent[ry] = rx
			else:
				parent[ry] = rx
				rank[rx] += 1
		
		num_to_index = {}
		for i, num in enumerate(nums):
			num_to_index[num] = i
		
		for d in range(1, threshold+1):
			L = multiples[d]
			if len(L) < 2:
				continue
			L.sort()
			rep = L[0]
			idx_rep = num_to_index[rep]
			for i in range(1, len(L)):
				x = L[i]
				if x * rep <= threshold * d:
					idx_x = num_to_index[x]
					union(idx_rep, idx_x)
				else:
					break
		
		roots = set()
		for i in range(n):
			roots.add(find(i))
		return len(roots)