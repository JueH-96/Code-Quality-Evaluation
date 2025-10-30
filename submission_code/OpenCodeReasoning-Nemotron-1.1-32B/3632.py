class Solution:
	def buttonWithLongestTime(self, events: List[List[int]]) -> int:
		prev_time = 0
		max_duration = 0
		candidate = None
		
		for event in events:
			idx = event[0]
			t = event[1]
			duration = t - prev_time
			if duration > max_duration:
				max_duration = duration
				candidate = idx
			elif duration == max_duration:
				if idx < candidate:
					candidate = idx
			prev_time = t
		
		return candidate