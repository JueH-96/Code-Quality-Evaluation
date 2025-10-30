from typing import List

class Solution:
	def longestSubsequence(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		max_len_arr = [[0] * 302 for _ in range(n)]
		for d in range(302):
			max_len_arr[0][d] = 1
		ans = 1
		for i in range(1, n):
			temp = [0] * 302
			temp[301] = 1
			for j in range(i):
				d_ij = abs(nums[i] - nums[j])
				candidate = max_len_arr[j][d_ij]
				new_length = candidate + 1
				if new_length > temp[d_ij]:
					temp[d_ij] = new_length
			m_arr = [0] * 302
			m_arr[301] = temp[301]
			for d in range(300, -1, -1):
				m_arr[d] = max(temp[d], m_arr[d+1])
			max_len_arr[i] = m_arr
			if m_arr[0] > ans:
				ans = m_arr[0]
		return ans