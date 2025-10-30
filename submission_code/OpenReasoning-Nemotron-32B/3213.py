class Solution:
	def countSubarrays(self, nums: List[int], k: int) -> int:
		n = len(nums)
		total_subarrays = n * (n + 1) // 2
		M = max(nums)
		A = [1 if x == M else 0 for x in nums]
		
		left = 0
		current = 0
		count_at_most = 0
		
		for right in range(n):
			current += A[right]
			while current > k - 1:
				current -= A[left]
				left += 1
			count_at_most += (right - left + 1)
		
		return total_subarrays - count_at_most