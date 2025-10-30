class Solution:
	def minSizeSubarray(self, nums: List[int], target: int) -> int:
		n = len(nums)
		total_single = sum(nums)
		total_double = total_single * 2
		arr = nums + nums
		
		def min_len_subarray_in_double(t):
			if t > total_double:
				return float('inf')
			left = 0
			cur_sum = 0
			min_len = float('inf')
			for right in range(len(arr)):
				cur_sum += arr[right]
				while cur_sum > t and left <= right:
					cur_sum -= arr[left]
					left += 1
				if cur_sum == t:
					min_len = min(min_len, right - left + 1)
			return min_len
		
		candidate = min_len_subarray_in_double(target)
		
		base = target // total_single
		rem = target % total_single
		
		if rem == 0:
			candidate = min(candidate, base * n)
		else:
			len_rem = min_len_subarray_in_double(rem)
			if len_rem != float('inf'):
				candidate = min(candidate, base * n + len_rem)
		
		if base >= 1:
			rem2 = rem + total_single
			len_rem2 = min_len_subarray_in_double(rem2)
			if len_rem2 != float('inf'):
				candidate = min(candidate, (base - 1) * n + len_rem2)
		
		return candidate if candidate != float('inf') else -1