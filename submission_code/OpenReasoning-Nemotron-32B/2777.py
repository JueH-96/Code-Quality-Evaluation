class Solution:
	def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
		n = len(nums)
		if n == 0:
			return []
		freq = [0] * 51
		suffix = [0] * n
		suffix[-1] = 0
		distinct = 0
		for i in range(n-2, -1, -1):
			num = nums[i+1]
			freq[num] += 1
			if freq[num] == 1:
				distinct += 1
			suffix[i] = distinct
		
		prefix_set = set()
		prefix_count = 0
		res = []
		for i in range(n):
			num = nums[i]
			if num not in prefix_set:
				prefix_set.add(num)
				prefix_count += 1
			res.append(prefix_count - suffix[i])
		
		return res