from typing import List

class Solution:
	def countDays(self, days: int, meetings: List[List[int]]) -> int:
		meetings.sort(key=lambda x: x[0])
		cur_start, cur_end = meetings[0]
		total_covered = 0
		n = len(meetings)
		for i in range(1, n):
			s, e = meetings[i]
			if s <= cur_end + 1:
				cur_end = max(cur_end, e)
			else:
				total_covered += cur_end - cur_start + 1
				cur_start, cur_end = s, e
		total_covered += cur_end - cur_start + 1
		return days - total_covered