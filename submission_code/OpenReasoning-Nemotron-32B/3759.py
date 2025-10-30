import heapq
from typing import List

class Solution:
	def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
		n = len(nums1)
		indices = list(range(n))
		indices.sort(key=lambda i: nums1[i])
		
		heap = []
		total = 0
		ans = [0] * n
		
		i = 0
		while i < n:
			j = i
			current_val = nums1[indices[i]]
			while j < n and nums1[indices[j]] == current_val:
				j += 1
			
			for idx in indices[i:j]:
				ans[idx] = total
			
			for idx in indices[i:j]:
				if len(heap) < k:
					heapq.heappush(heap, nums2[idx])
					total += nums2[idx]
				else:
					if nums2[idx] > heap[0]:
						popped = heapq.heappop(heap)
						total -= popped
						heapq.heappush(heap, nums2[idx])
						total += nums2[idx]
			
			i = j
		
		return ans