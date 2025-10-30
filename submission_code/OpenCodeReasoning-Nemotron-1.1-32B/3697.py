import heapq
import math
import itertools
from typing import List

class Solution:
	def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
		n = len(target)
		subset_info = {}
		
		def compute_lcm(arr):
			if not arr:
				return 1
			l = arr[0]
			for i in range(1, len(arr)):
				g = math.gcd(l, arr[i])
				l = l // g * arr[i]
			return l
		
		for mask in range(1, 1 << n):
			arr = []
			for i in range(n):
				if mask & (1 << i):
					arr.append(target[i])
			L = compute_lcm(arr)
			heap = []
			for idx, a in enumerate(nums):
				r = a % L
				if r == 0:
					cost = 0
				else:
					cost = L - r
				if len(heap) < 4:
					heapq.heappush(heap, (-cost, idx))
				else:
					if cost < -heap[0][0]:
						heapq.heappop(heap)
						heapq.heappush(heap, (-cost, idx))
			candidate_list = [(-c, idx) for c, idx in heap]
			subset_info[mask] = candidate_list
		
		def gen_partitions(elems):
			if len(elems) == 0:
				yield []
			else:
				first = elems[0]
				for smaller in gen_partitions(elems[1:]):
					for i in range(len(smaller)):
						new_part = smaller.copy()
						new_part[i] = new_part[i] + [first]
						yield new_part
					yield smaller + [[first]]
		
		indices_list = list(range(n))
		min_total = float('inf')
		
		for partition in gen_partitions(indices_list):
			sets = []
			for subset in partition:
				mask_sub = 0
				for i in subset:
					mask_sub |= (1 << i)
				sets.append(mask_sub)
			candidate_lists = [subset_info[mask] for mask in sets]
			k = len(sets)
			best_for_partition = float('inf')
			for choice in itertools.product(*candidate_lists):
				indices_used = [idx for cost, idx in choice]
				if len(set(indices_used)) == k:
					total_cost = sum(cost for cost, idx in choice)
					if total_cost < best_for_partition:
						best_for_partition = total_cost
			if best_for_partition < min_total:
				min_total = best_for_partition
				
		return min_total