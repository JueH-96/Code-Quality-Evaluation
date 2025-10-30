class Solution:
	def maximumSubarraySum(self, nums: List[int], k: int) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + nums[i - 1]
		
		min_prefix = {}
		ans = None
		
		for j in range(n):
			t1 = nums[j] - k
			t2 = nums[j] + k
			
			if t1 in min_prefix:
				total = prefix[j + 1] - min_prefix[t1]
				if ans is None or total > ans:
					ans = total
			if t2 in min_prefix:
				total = prefix[j + 1] - min_prefix[t2]
				if ans is None or total > ans:
					ans = total
			
			current_prefix = prefix[j]
			if nums[j] in min_prefix:
				if current_prefix < min_prefix[nums[j]]:
					min_prefix[nums[j]] = current_prefix
			else:
				min_prefix[nums[j]] = current_prefix
		
		return ans if ans is not None else 0