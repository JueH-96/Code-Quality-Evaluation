class Solution:
	def minDifference(self, nums: List[int]) -> int:
		n = len(nums)
		
		def check(D):
			for i in range(n-1):
				if nums[i] != -1 and nums[i+1] != -1:
					if abs(nums[i] - nums[i+1]) > D:
						return False
			
			intervals = []
			for i in range(n):
				if nums[i] == -1:
					low_bound = -10**18
					high_bound = 10**18
					if i-1 >= 0 and nums[i-1] != -1:
						low_bound = max(low_bound, nums[i-1] - D)
						high_bound = min(high_bound, nums[i-1] + D)
					if i+1 < n and nums[i+1] != -1:
						low_bound = max(low_bound, nums[i+1] - D)
						high_bound = min(high_bound, nums[i+1] + D)
					if low_bound > high_bound:
						return False
					intervals.append((low_bound, high_bound))
			
			if not intervals:
				return True
				
			intervals.sort(key=lambda x: x[1])
			
			p1 = intervals[0][1]
			covered_all = True
			for inter in intervals:
				if not (inter[0] <= p1 <= inter[1]):
					covered_all = False
					break
					
			if covered_all:
				return True
				
			remaining = []
			for inter in intervals:
				if not (inter[0] <= p1 <= inter[1]):
					remaining.append(inter)
					
			if not remaining:
				return True
				
			low_remaining = -10**18
			high_remaining = 10**18
			for inter in remaining:
				low_remaining = max(low_remaining, inter[0])
				high_remaining = min(high_remaining, inter[1])
				
			if low_remaining > high_remaining:
				return False
				
			if max(low_remaining, p1 - D) <= min(high_remaining, p1 + D):
				return True
				
			return False
		
		low = 0
		high = 10**9
		while low < high:
			mid = (low + high) // 2
			if check(mid):
				high = mid
			else:
				low = mid + 1
				
		return low