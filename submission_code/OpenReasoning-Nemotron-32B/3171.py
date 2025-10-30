class Solution:
	def minSum(self, nums1: List[int], nums2: List[int]) -> int:
		total1 = 0
		count1 = 0
		for num in nums1:
			if num == 0:
				count1 += 1
			else:
				total1 += num
		
		total2 = 0
		count2 = 0
		for num in nums2:
			if num == 0:
				count2 += 1
			else:
				total2 += num
		
		min1 = total1 + count1
		min2 = total2 + count2
		
		if count1 == 0 and total1 < min2:
			return -1
		if count2 == 0 and total2 < min1:
			return -1
		
		return max(min1, min2)