from collections import defaultdict

class Solution:
	def minimumSeconds(self, nums: List[int]) -> int:
		n = len(nums)
		if all(x == nums[0] for x in nums):
			return 0
		
		pos_dict = defaultdict(list)
		for i, num in enumerate(nums):
			pos_dict[num].append(i)
		
		ans = float('inf')
		for x, positions in pos_dict.items():
			m = len(positions)
			gaps = []
			for i in range(m):
				if i < m - 1:
					gap = positions[i+1] - positions[i] - 1
				else:
					gap = n - positions[-1] + positions[0] - 1
				gaps.append(gap)
			
			max_time = 0
			for gap in gaps:
				t = (gap + 1) // 2
				if t > max_time:
					max_time = t
			if max_time < ans:
				ans = max_time
		
		return ans