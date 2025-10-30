from collections import defaultdict
from typing import List

class Solution:
	def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
		n = len(nums)
		if n == 0:
			return []
		arr = [(nums[i], i) for i in range(n)]
		arr.sort(key=lambda x: x[0])
		
		parent = list(range(n))
		rank = [0] * n
		
		def find(x: int) -> int:
			if parent[x] != x:
				parent[x] = find(parent[x])
			return parent[x]
		
		def union(x: int, y: int):
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
		
		for i in range(1, n):
			if arr[i][0] - arr[i-1][0] <= limit:
				idx1 = arr[i-1][1]
				idx2 = arr[i][1]
				union(idx1, idx2)
		
		components = defaultdict(list)
		for i in range(n):
			root = find(i)
			components[root].append(i)
		
		res = [0] * n
		for comp in components.values():
			indices = sorted(comp)
			vals = sorted(nums[i] for i in comp)
			for idx, val in zip(indices, vals):
				res[idx] = val
				
		return res