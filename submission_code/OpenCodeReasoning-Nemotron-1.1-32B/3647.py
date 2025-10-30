import heapq

class Solution:
	def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
		n = len(nums)
		total_cover = [0] * (n + 2)
		for l, r in queries:
			total_cover[l] += 1
			if r + 1 < n + 2:
				total_cover[r + 1] -= 1
		
		cur_cover = 0
		for i in range(n):
			cur_cover += total_cover[i]
			if cur_cover < nums[i]:
				return -1
		
		events_start = [[] for _ in range(n)]
		for l, r in queries:
			events_start[l].append(r)
		
		heap = []
		diff = [0] * (n + 2)
		cur = 0
		total_used = 0
		
		for i in range(n):
			for r_val in events_start[i]:
				heapq.heappush(heap, -r_val)
			
			while heap and -heap[0] < i:
				heapq.heappop(heap)
			
			cur += diff[i]
			deficit = nums[i] - cur
			used_here = 0
			while deficit > 0 and heap:
				r_val = -heapq.heappop(heap)
				diff[i] += 1
				if r_val + 1 < n:
					diff[r_val + 1] -= 1
				cur += 1
				deficit -= 1
				used_here += 1
			
			if deficit > 0:
				return -1
			total_used += used_here
		
		return len(queries) - total_used