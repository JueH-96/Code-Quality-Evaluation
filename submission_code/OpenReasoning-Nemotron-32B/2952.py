class Solution:
	def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
		sum_nums1 = sum(nums1)
		sum_nums2 = sum(nums2)
		n = len(nums1)
		if sum_nums1 <= x:
			return 0
		
		arr = [(nums2[i], nums1[i]) for i in range(n)]
		arr.sort(key=lambda val: (-val[0], -val[1]))
		
		for T in range(1, n + 1):
			k = min(n, T)
			R = 0
			for i in range(k):
				R += arr[i][1] + (T - i) * arr[i][0]
			total_min = T * sum_nums2 + sum_nums1 - R
			if total_min <= x:
				return T
		
		return -1