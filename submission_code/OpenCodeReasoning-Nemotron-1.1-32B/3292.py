class Solution:
	def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
		n = len(nums)
		m = len(changeIndices)
		last_occurrence = [-1] * n
		count = 0
		ans = -1
		for T in range(1, m + 1):
			s_index = T - 1
			i = changeIndices[s_index] - 1
			if last_occurrence[i] == -1:
				count += 1
			last_occurrence[i] = s_index
			
			if count < n:
				continue
				
			arr = []
			for i in range(n):
				arr.append((last_occurrence[i], nums[i]))
			arr.sort(key=lambda x: x[0])
			
			time_used = 0
			valid = True
			for idx, (last_i, num_i) in enumerate(arr):
				time_used += num_i
				if time_used > last_i - idx:
					valid = False
					break
					
			if valid:
				ans = T
				break
				
		return ans