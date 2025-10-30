from typing import List
import heapq
import bisect

class Solution:
	def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
		INF = 10**18
		n = len(fruits)
		unique_caps = sorted(set(baskets))
		cap_to_index = {cap: idx for idx, cap in enumerate(unique_caps)}
		m = len(unique_caps)
		
		heaps = {}
		for cap in unique_caps:
			heaps[cap] = []
		
		for idx, cap in enumerate(baskets):
			heapq.heappush(heaps[cap], idx)
		
		arr = [INF] * m
		for cap in unique_caps:
			if heaps[cap]:
				arr[cap_to_index[cap]] = heaps[cap][0]
		
		class SegmentTree:
			def __init__(self, data):
				self.n = len(data)
				self.size = 1
				while self.size < self.n:
					self.size *= 2
				self.tree = [INF] * (2 * self.size)
				for i in range(self.n):
					self.tree[self.size + i] = data[i]
				for i in range(self.size - 1, 0, -1):
					self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
			
			def update(self, idx, value):
				pos = self.size + idx
				self.tree[pos] = value
				pos //= 2
				while pos:
					self.tree[pos] = min(self.tree[2*pos], self.tree[2*pos+1])
					pos //= 2
			
			def query(self, l, r):
				l += self.size
				r += self.size
				res = INF
				while l <= r:
					if l % 2 == 1:
						res = min(res, self.tree[l])
						l += 1
					if r % 2 == 0:
						res = min(res, self.tree[r])
						r -= 1
					l //= 2
					r //= 2
				return res
		
		seg_tree = SegmentTree(arr)
		unplaced = 0
		
		for x in fruits:
			L = bisect.bisect_left(unique_caps, x)
			if L == m:
				unplaced += 1
			else:
				min_idx = seg_tree.query(L, m-1)
				if min_idx == INF:
					unplaced += 1
				else:
					cap_used = baskets[min_idx]
					heapq.heappop(heaps[cap_used])
					if heaps[cap_used]:
						new_top = heaps[cap_used][0]
					else:
						new_top = INF
					seg_tree.update(cap_to_index[cap_used], new_top)
		
		return unplaced