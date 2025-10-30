from collections import defaultdict

class Solution:
	def maxSum(self, nums: List[int]) -> int:
		groups = defaultdict(list)
		for num in nums:
			max_digit = max(str(num))
			groups[max_digit].append(num)
		
		ans = -1
		for lst in groups.values():
			if len(lst) < 2:
				continue
			a = b = 0
			for x in lst:
				if x > a:
					b = a
					a = x
				elif x > b:
					b = x
			total = a + b
			if total > ans:
				ans = total
		return ans