class Solution:
	def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
		n = len(nums1)
		cand1 = 0
		lastA1 = nums1[-1]
		lastB1 = nums2[-1]
		valid1 = True
		for i in range(n - 1):
			if nums1[i] <= lastA1 and nums2[i] <= lastB1:
				continue
			elif nums2[i] <= lastA1 and nums1[i] <= lastB1:
				cand1 += 1
			else:
				valid1 = False
				break
		
		cand2 = 1
		lastA2 = nums2[-1]
		lastB2 = nums1[-1]
		valid2 = True
		for i in range(n - 1):
			if nums1[i] <= lastA2 and nums2[i] <= lastB2:
				continue
			elif nums2[i] <= lastA2 and nums1[i] <= lastB2:
				cand2 += 1
			else:
				valid2 = False
				break
		
		if not valid1 and not valid2:
			return -1
		if not valid1:
			return cand2
		if not valid2:
			return cand1
		return min(cand1, cand2)