class Solution:
	def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
		n = len(nums)
		m = len(changeIndices)
		for s in range(1, m + 1):
			last_occurrence = [-1] * n
			for t in range(s):
				idx = changeIndices[t] - 1
				last_occurrence[idx] = t
			
			if any(x == -1 for x in last_occurrence):
				continue
			
			events = []
			for i in range(n):
				events.append((last_occurrence[i], nums[i]))
			events.sort(key=lambda x: x[0])
			
			total_req_so_far = 0
			total_processed = 0
			i = 0
			valid = True
			while i < n:
				j = i
				d = events[i][0]
				group_req = 0
				while j < n and events[j][0] == d:
					group_req += events[j][1]
					j += 1
				total_req_so_far += group_req
				if d - total_processed < total_req_so_far:
					valid = False
					break
				total_processed = j
				i = j
			
			if valid:
				return s
		
		return -1