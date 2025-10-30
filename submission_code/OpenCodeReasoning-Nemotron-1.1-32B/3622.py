class Solution:
	def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
		if not nums:
			return 0
		
		min_val = min(nums) - k
		max_val = max(nums) + k
		size = max_val - min_val + 1
		
		if size <= 0:
			return 0
		
		freq_arr = [0] * size
		for num in nums:
			idx = num - min_val
			if 0 <= idx < size:
				freq_arr[idx] += 1
		
		count = 0
		start = min_val
		end = min_val + k
		if end > max_val:
			end = max_val
		for val in range(start, end + 1):
			idx_val = val - min_val
			if 0 <= idx_val < size:
				count += freq_arr[idx_val]
		
		best_freq = 0
		for x in range(min_val, max_val + 1):
			idx_x = x - min_val
			base = freq_arr[idx_x] if 0 <= idx_x < size else 0
			total_freq = base + min(numOperations, count - base)
			if total_freq > best_freq:
				best_freq = total_freq
			
			remove_val = x - k
			if min_val <= remove_val <= max_val:
				idx_remove = remove_val - min_val
				if 0 <= idx_remove < size:
					count -= freq_arr[idx_remove]
			
			add_val = x + k + 1
			if min_val <= add_val <= max_val:
				idx_add = add_val - min_val
				if 0 <= idx_add < size:
					count += freq_arr[idx_add]
		
		return best_freq