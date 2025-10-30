class Solution:
	def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
		n = len(nums)
		prefix_arr = [0] * n
		seen_pre = set()
		for i in range(n):
			seen_pre.add(nums[i])
			prefix_arr[i] = len(seen_pre)
		
		suffix_arr = [0] * (n + 1)
		seen_suf = set()
		for i in range(n - 1, -1, -1):
			seen_suf.add(nums[i])
			suffix_arr[i] = len(seen_suf)
		
		res = []
		for i in range(n):
			res.append(prefix_arr[i] - suffix_arr[i + 1])
		
		return res