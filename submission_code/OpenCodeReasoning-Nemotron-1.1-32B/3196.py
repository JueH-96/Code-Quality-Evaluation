class Solution:
	def maxFrequencyScore(self, nums: List[int], k: int) -> int:
		n = len(nums)
		arr = sorted(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + arr[i - 1]
		
		low, high = 1, n
		ans = 1
		
		def check(f):
			for l in range(n - f + 1):
				r = l + f - 1
				mid_index = l + (f - 1) // 2
				left_count = mid_index - l + 1
				left_sum = left_count * arr[mid_index] - (prefix[mid_index + 1] - prefix[l])
				
				right_count = r - mid_index
				right_sum = (prefix[r + 1] - prefix[mid_index + 1]) - right_count * arr[mid_index]
				
				total_cost = left_sum + right_sum
				if total_cost <= k:
					return True
			return False
		
		while low <= high:
			mid_f = (low + high) // 2
			if check(mid_f):
				ans = mid_f
				low = mid_f + 1
			else:
				high = mid_f - 1
				
		return ans