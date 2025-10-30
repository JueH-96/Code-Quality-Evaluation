class Solution:
	def incremovableSubarrayCount(self, nums: List[int]) -> int:
		n = len(nums)
		prefix_ok = [False] * (n + 1)
		prefix_ok[0] = True
		if n >= 1:
			prefix_ok[1] = True
			for i in range(2, n + 1):
				prefix_ok[i] = prefix_ok[i - 1] and (nums[i - 1] > nums[i - 2])
		
		suffix_ok = [False] * (n + 1)
		suffix_ok[n] = True
		if n >= 1:
			suffix_ok[n - 1] = True
			for i in range(n - 2, -1, -1):
				suffix_ok[i] = (nums[i] < nums[i + 1]) and suffix_ok[i + 1]
		
		count = 0
		for l in range(n):
			for r in range(l, n):
				if l == 0:
					if suffix_ok[r + 1]:
						count += 1
				elif r == n - 1:
					if prefix_ok[l]:
						count += 1
				else:
					if prefix_ok[l] and suffix_ok[r + 1] and (nums[l - 1] < nums[r + 1]):
						count += 1
		return count