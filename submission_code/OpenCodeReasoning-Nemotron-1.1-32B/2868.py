from collections import deque

class Solution:
	def continuousSubarrays(self, nums: List[int]) -> int:
		min_deque = deque()
		max_deque = deque()
		l = 0
		ans = 0
		n = len(nums)
		
		for r in range(n):
			while min_deque and nums[min_deque[-1]] > nums[r]:
				min_deque.pop()
			min_deque.append(r)
			
			while max_deque and nums[max_deque[-1]] < nums[r]:
				max_deque.pop()
			max_deque.append(r)
			
			while nums[max_deque[0]] - nums[min_deque[0]] > 2:
				if min_deque[0] == l:
					min_deque.popleft()
				if max_deque[0] == l:
					max_deque.popleft()
				l += 1
				
			ans += (r - l + 1)
			
		return ans