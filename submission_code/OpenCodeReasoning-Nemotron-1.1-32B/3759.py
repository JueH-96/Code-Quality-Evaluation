import heapq

class Solution:
	def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
		n = len(nums1)
		ans = [0] * n
		indices = sorted(range(n), key=lambda i: nums1[i])
		
		heap = []
		total_sum = 0
		i = 0
		while i < n:
			j = i
			while j < n and nums1[indices[j]] == nums1[indices[i]]:
				j += 1
			for idx in range(i, j):
				orig_index = indices[idx]
				ans[orig_index] = total_sum
			for idx in range(i, j):
				orig_index = indices[idx]
				num = nums2[orig_index]
				if len(heap) < k:
					heapq.heappush(heap, num)
					total_sum += num
				else:
					if num > heap[0]:
						removed = heapq.heappop(heap)
						total_sum -= removed
						heapq.heappush(heap, num)
						total_sum += num
			i = j
		return ans