class Solution:
	def buttonWithLongestTime(self, events: List[List[int]]) -> int:
		prev_time = 0
		max_duration = 0
		result_index = None
		
		for idx, t in events:
			duration = t - prev_time
			prev_time = t
			
			if duration > max_duration:
				max_duration = duration
				result_index = idx
			elif duration == max_duration:
				if result_index is None:
					result_index = idx
				elif idx < result_index:
					result_index = idx
		
		return result_index