class Solution:
	def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
		n = len(nums1)
		# Case 1: no swap at the last index
		x1 = nums1[-1]
		y1 = nums2[-1]
		swaps1 = 0
		valid1 = True
		for i in range(n-1):
			a, b = nums1[i], nums2[i]
			if a <= x1 and b <= y1:
				continue
			elif b <= x1 and a <= y1:
				swaps1 += 1
			else:
				valid1 = False
				break
		
		# Case 2: swap at the last index
		x2 = nums2[-1]
		y2 = nums1[-1]
		swaps2 = 1
		valid2 = True
		for i in range(n-1):
			a, b = nums1[i], nums2[i]
			if a <= x2 and b <= y2:
				continue
			elif b <= x2 and a <= y2:
				swaps2 += 1
			else:
				valid2 = False
				break
		
		if not valid1 and not valid2:
			return -1
		elif valid1 and valid2:
			return min(swaps1, swaps2)
		elif valid1:
			return swaps1
		else:
			return swaps2