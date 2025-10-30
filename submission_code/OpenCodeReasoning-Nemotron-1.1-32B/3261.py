class Solution:
	def minOrAfterOperations(self, nums: List[int], k: int) -> int:
		n = len(nums)
		m = n - k
		total_or = 0
		for x in nums:
			total_or |= x
		
		counts = [0] * 31
		for x in nums:
			for bit in range(31):
				if (x >> bit) & 1 == 0:
					counts[bit] += 1
		
		ans = 0
		for bit in range(31):
			if (total_or >> bit) & 1:
				if counts[bit] < m:
					ans |= (1 << bit)
		return ans