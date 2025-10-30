class Solution:
	def medianOfUniquenessArray(self, nums: List[int]) -> int:
		n = len(nums)
		total_subarrays = n * (n + 1) // 2
		k = (total_subarrays + 1) // 2
		
		low, high = 1, n
		
		def count_le(x):
			max_val = 100000
			freq = [0] * (max_val + 1)
			distinct = 0
			r = -1
			count = 0
			for l in range(n):
				while r + 1 < n:
					nxt = nums[r + 1]
					if freq[nxt] == 0:
						if distinct == x:
							break
						distinct += 1
					r += 1
					freq[nxt] += 1
				
				count += (r - l + 1)
				
				freq[nums[l]] -= 1
				if freq[nums[l]] == 0:
					distinct -= 1
					
			return count
		
		while low < high:
			mid = (low + high) // 2
			if count_le(mid) >= k:
				high = mid
			else:
				low = mid + 1
				
		return low