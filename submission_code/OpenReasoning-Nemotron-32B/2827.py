from typing import List
import math
from collections import defaultdict

class Solution:
	def canTraverseAllPairs(self, nums: List[int]) -> bool:
		n = len(nums)
		if n == 1:
			return True
		if any(num == 1 for num in nums):
			return False
		
		max_num = max(nums)
		spf = list(range(max_num + 1))
		sqrt_max = int(math.isqrt(max_num))
		for i in range(2, sqrt_max + 1):
			if spf[i] == i:
				for j in range(i * i, max_num + 1, i):
					if spf[j] == j:
						spf[j] = i
		
		class DSU:
			def __init__(self, n):
				self.parent = list(range(n))
				self.rank = [0] * n
			
			def find(self, x):
				if self.parent[x] != x:
					self.parent[x] = self.find(self.parent[x])
				return self.parent[x]
			
			def union(self, x, y):
				rx = self.find(x)
				ry = self.find(y)
				if rx == ry:
					return
				if self.rank[rx] < self.rank[ry]:
					rx, ry = ry, rx
				self.parent[ry] = rx
				if self.rank[rx] == self.rank[ry]:
					self.rank[rx] += 1
		
		dsu = DSU(n)
		prime_groups = defaultdict(list)
		
		for i, num in enumerate(nums):
			temp = num
			while temp > 1:
				p = spf[temp]
				prime_groups[p].append(i)
				while temp % p == 0:
					temp //= p
		
		for indices in prime_groups.values():
			if len(indices) > 1:
				base = indices[0]
				for j in range(1, len(indices)):
					dsu.union(base, indices[j])
		
		roots = set()
		for i in range(n):
			roots.add(dsu.find(i))
		return len(roots) == 1