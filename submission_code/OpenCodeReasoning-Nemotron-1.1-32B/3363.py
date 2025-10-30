import heapq
from collections import defaultdict

class Solution:
	def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
		count = defaultdict(int)
		freq_to_count = defaultdict(int)
		heap = []
		ans = []
		
		for i in range(len(nums)):
			num = nums[i]
			d = freq[i]
			
			old_freq = count[num]
			new_freq = old_freq + d
			count[num] = new_freq
			
			if old_freq > 0:
				freq_to_count[old_freq] -= 1
			if new_freq > 0:
				freq_to_count[new_freq] += 1
				
			heapq.heappush(heap, -new_freq)
			
			while heap:
				candidate = -heap[0]
				if candidate in freq_to_count and freq_to_count[candidate] > 0:
					break
				heapq.heappop(heap)
				
			if heap:
				ans.append(candidate)
			else:
				ans.append(0)
				
		return ans