from typing import List

class Solution:
	def countDays(self, days: int, meetings: List[List[int]]) -> int:
		meetings.sort(key=lambda x: x[0])
		total_meeting_days = 0
		cur_start, cur_end = meetings[0]
		for i in range(1, len(meetings)):
			s, e = meetings[i]
			if s <= cur_end:
				cur_end = max(cur_end, e)
			else:
				total_meeting_days += cur_end - cur_start + 1
				cur_start, cur_end = s, e
		total_meeting_days += cur_end - cur_start + 1
		return days - total_meeting_days