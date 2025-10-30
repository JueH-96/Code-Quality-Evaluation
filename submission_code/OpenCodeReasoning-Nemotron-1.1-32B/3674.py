class Solution:
	def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
		n = len(nums)
		ans = 0
		for l in range(n):
			current_max = nums[l]
			total_ops = 0
			r = l
			while r < n:
				if nums[r] < current_max:
					total_ops += current_max - nums[r]
				else:
					current_max = nums[r]
				if total_ops > k:
					break
				r += 1
			ans += (r - l)
		return ans