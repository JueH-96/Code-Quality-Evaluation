class Solution:
	def medianOfUniquenessArray(self, nums: List[int]) -> int:
		n = len(nums)
		total_subarrays = n * (n + 1) // 2
		k = (total_subarrays - 1) // 2
		
		def count_subarrays(x):
			freq = [0] * 100002
			l = 0
			distinct = 0
			total_count = 0
			for r in range(n):
				num = nums[r]
				if freq[num] == 0:
					distinct += 1
				freq[num] += 1
				
				while distinct > x and l <= r:
					left_num = nums[l]
					freq[left_num] -= 1
					if freq[left_num] == 0:
						distinct -= 1
					l += 1
					
				total_count += (r - l + 1)
			return total_count
		
		low, high = 1, n
		while low < high:
			mid = (low + high) // 2
			if count_subarrays(mid) >= k + 1:
				high = mid
			else:
				low = mid + 1
				
		return low