from typing import List

class Solution:
	def minimumDifference(self, nums: List[int], k: int) -> int:
		n = len(nums)
		LOG = n.bit_length()
		st = [[0] * n for _ in range(LOG)]
		for i in range(n):
			st[0][i] = nums[i]
		
		for j in range(1, LOG):
			step = 1 << (j-1)
			for i in range(0, n - (1 << j) + 1):
				st[j][i] = st[j-1][i] | st[j-1][i+step]
		
		def query(l, r):
			length = r - l + 1
			j = length.bit_length() - 1
			return st[j][l] | st[j][r - (1 << j) + 1]
		
		ans = 10**18
		total_OR = query(0, n-1)
		if total_OR < k:
			for l in range(n):
				or_val = query(l, n-1)
				ans = min(ans, abs(k - or_val))
			return ans
		
		for l in range(n):
			low, high = l, n-1
			r_index = -1
			while low <= high:
				mid = (low + high) // 2
				or_val = query(l, mid)
				if or_val >= k:
					r_index = mid
					high = mid - 1
				else:
					low = mid + 1
					
			if r_index != -1:
				or1 = query(l, r_index)
				if or1 == k:
					return 0
				ans = min(ans, abs(k - or1))
				if r_index > l:
					or2 = query(l, r_index-1)
					if or2 == k:
						return 0
					ans = min(ans, abs(k - or2))
			else:
				or_val = query(l, n-1)
				ans = min(ans, abs(k - or_val))
				
		return ans