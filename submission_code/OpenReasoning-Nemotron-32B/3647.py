import heapq
from typing import List

class Solution:
	def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
		n = len(nums)
		events = [[] for _ in range(n)]
		for l, r in queries:
			if l < n:
				events[l].append(r)
		
		class Fenw:
			def __init__(self, size):
				self.n = size
				self.tree = [0] * (self.n + 1)
			
			def update(self, index, delta):
				i = index + 1
				while i <= self.n:
					self.tree[i] += delta
					i += i & -i
			
			def query(self, index):
				s = 0
				i = index + 1
				while i:
					s += self.tree[i]
					i -= i & -i
				return s
		
		bit = Fenw(n + 1)
		heap = []
		kept = 0
		
		for i in range(n):
			while heap and -heap[0] < i:
				heapq.heappop(heap)
			
			for r in events[i]:
				heapq.heappush(heap, -r)
			
			cur = bit.query(i)
			while cur < nums[i]:
				if not heap:
					return -1
				r_max = -heapq.heappop(heap)
				bit.update(i, 1)
				if r_max + 1 <= n:
					bit.update(r_max + 1, -1)
				cur += 1
				kept += 1
		
		return len(queries) - kept