class Solution:
	def countSubarrays(self, nums: List[int], k: int) -> int:
		total = 0
		n = len(nums)
		U_list = []
		for b in range(32):
			if (k >> b) & 1 == 0:
				U_list.append(b)
		
		i = 0
		while i < n:
			if (nums[i] & k) == k:
				j = i
				while j < n and (nums[j] & k) == k:
					j += 1
				seg_start = i
				seg_end = j - 1
				seg_len = j - i
				
				if len(U_list) == 0:
					total += seg_len * (seg_len + 1) // 2
				else:
					last = [-1] * len(U_list)
					for idx in range(seg_start, seg_end + 1):
						for b_idx, b in enumerate(U_list):
							if (nums[idx] >> b) & 1 == 0:
								last[b_idx] = idx
						if min(last) == -1:
							continue
						else:
							l0 = min(last)
							total += (l0 - seg_start + 1)
				i = j
			else:
				i += 1
				
		return total