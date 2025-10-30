class Solution:
	def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
		n = len(properties)
		if n == 0:
			return 0
		
		sets = [set(p) for p in properties]
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
		
		for i in range(n):
			for j in range(i + 1, n):
				common = len(sets[i] & sets[j])
				if common >= k:
					union(i, j)
		
		roots = set()
		for i in range(n):
			roots.add(find(i))
		return len(roots)