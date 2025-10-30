class Solution:
	def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
		n = len(nums1)
		base0 = sum(nums1)
		S2 = sum(nums2)
		arr = list(zip(nums1, nums2))
		arr.sort(key=lambda x: x[1], reverse=True)
		
		for t in range(0, n + 1):
			base = base0 + t * S2
			k_max = min(t, n)
			if k_max == 0:
				max_saving = 0
			else:
				dp = [-10**18] * (k_max + 1)
				dp[0] = 0
				for a, b in arr:
					for k in range(k_max, 0, -1):
						s_val = t - k + 1
						candidate = dp[k - 1] + a + s_val * b
						if candidate > dp[k]:
							dp[k] = candidate
				max_saving = max(dp)
			total_min = base - max_saving
			if total_min <= x:
				return t
		return -1