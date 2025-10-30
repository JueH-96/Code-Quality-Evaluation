class Solution:
	def minSum(self, nums1: List[int], nums2: List[int]) -> int:
		s1 = sum(nums1)
		c1 = nums1.count(0)
		s2 = sum(nums2)
		c2 = nums2.count(0)
		
		min_sum1 = s1 + c1
		min_sum2 = s2 + c2
		
		if c1 == 0 and c2 == 0:
			return s1 if s1 == s2 else -1
		
		if c1 == 0:
			if s1 < min_sum2:
				return -1
			return s1
		
		if c2 == 0:
			if s2 < min_sum1:
				return -1
			return s2
		
		return max(min_sum1, min_sum2)