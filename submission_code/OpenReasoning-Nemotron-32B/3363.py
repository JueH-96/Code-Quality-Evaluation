import heapq
from collections import defaultdict

class Solution:
	def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
		current_freq = defaultdict(int)
		heap = []
		n = len(nums)
		ans = [0] * n
		
		for i in range(n):
			id_val = nums[i]
			delta = freq[i]
			
			old = current_freq[id_val]
			new = old + delta
			current_freq[id_val] = new
			
			heapq.heappush(heap, (-new, id_val))
			
			while heap:
				neg_freq, top_id = heap[0]
				real_freq = -neg_freq
				if current_freq[top_id] != real_freq:
					heapq.heappop(heap)
				else:
					break
					
			if heap:
				ans[i] = -heap[0][0]
			else:
				ans[i] = 0
				
		return ans