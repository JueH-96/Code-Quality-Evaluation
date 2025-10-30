from collections import deque

class Solution:
	def continuousSubarrays(self, nums: List[int]) -> int:
		n = len(nums)
		min_deque = deque()
		max_deque = deque()
		r = 0
		total = 0
		
		for l in range(n):
			while r < n:
				current_min = nums[min_deque[0]] if min_deque else nums[r]
				current_max = nums[max_deque[0]] if max_deque else nums[r]
				new_min = min(current_min, nums[r])
				new_max = max(current_max, nums[r])
				if new_max - new_min <= 2:
					while min_deque and nums[min_deque[-1]] > nums[r]:
						min_deque.pop()
					min_deque.append(r)
					while max_deque and nums[max_deque[-1]] < nums[r]:
						max_deque.pop()
					max_deque.append(r)
					r += 1
				else:
					break
					
			total += r - l
			
			if min_deque and min_deque[0] == l:
				min_deque.popleft()
			if max_deque and max_deque[0] == l:
				max_deque.popleft()
				
		return total