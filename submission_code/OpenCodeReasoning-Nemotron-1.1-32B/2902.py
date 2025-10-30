class Solution:
	def maxSum(self, nums: List[int]) -> int:
		groups = {}
		for num in nums:
			s = str(num)
			max_digit = int(max(s))
			if max_digit not in groups:
				groups[max_digit] = []
			groups[max_digit].append(num)
		
		best = -1
		for key in groups:
			arr = groups[key]
			if len(arr) < 2:
				continue
			first = -1
			second = -1
			for x in arr:
				if x > first:
					second = first
					first = x
				elif x > second:
					second = x
			total = first + second
			if total > best:
				best = total
		return best