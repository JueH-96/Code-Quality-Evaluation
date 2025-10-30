import bisect

class Solution:
	def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
		max_val = 50000
		n = len(nums)
		freq_num = [0] * (max_val + 1)
		for num in nums:
			if num <= max_val:
				freq_num[num] += 1
		
		cnt = [0] * (max_val + 1)
		for d in range(1, max_val + 1):
			for multiple in range(d, max_val + 1, d):
				cnt[d] += freq_num[multiple]
				
		f = [0] * (max_val + 1)
		for d in range(max_val, 0, -1):
			total = cnt[d] * (cnt[d] - 1) // 2
			k = 2
			while k * d <= max_val:
				total -= f[k * d]
				k += 1
			f[d] = total
			
		prefix = [0] * (max_val + 1)
		for d in range(1, max_val + 1):
			prefix[d] = prefix[d - 1] + f[d]
			
		ans = []
		for q in queries:
			d_val = bisect.bisect_right(prefix, q)
			ans.append(d_val)
			
		return ans