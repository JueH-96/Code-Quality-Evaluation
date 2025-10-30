from typing import List

class Solution:
	def maximumBeauty(self, nums: List[int], k: int) -> int:
		events = []
		for num in nums:
			l = num - k
			r = num + k
			events.append((l, 1))
			events.append((r + 1, -1))
		
		events.sort(key=lambda x: x[0])
		
		count = 0
		max_count = 0
		i = 0
		n = len(events)
		while i < n:
			current_coord = events[i][0]
			j = i
			while j < n and events[j][0] == current_coord:
				count += events[j][1]
				j += 1
			if count > max_count:
				max_count = count
			i = j
			
		return max_count