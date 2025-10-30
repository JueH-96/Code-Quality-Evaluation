class Solution:
	def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
		n = len(startTime)
		meetings = sorted(zip(startTime, endTime))
		startTime_sorted = [m[0] for m in meetings]
		endTime_sorted = [m[1] for m in meetings]
		
		def check(X):
			events = []
			for i in range(n):
				L = startTime_sorted[i] - X + 1
				R = endTime_sorted[i] - 1
				if L > R:
					continue
				start_event = max(L, 0)
				end_event = min(R, eventTime - X)
				if start_event > end_event:
					continue
				events.append((start_event, 1))
				events.append((end_event + 1, -1))
			
			if not events:
				return True
			events.sort(key=lambda x: x[0])
			current_count = 0
			min_count = float('inf')
			ptr = 0
			current_pos = 0
			while ptr < len(events) and events[ptr][0] <= eventTime - X:
				pos = events[ptr][0]
				if current_pos < pos:
					min_count = min(min_count, current_count)
				while ptr < len(events) and events[ptr][0] == pos:
					current_count += events[ptr][1]
					ptr += 1
				current_pos = pos
			if current_pos <= eventTime - X:
				min_count = min(min_count, current_count)
			return min_count <= k

		lo, hi = 0, eventTime
		ans = 0
		while lo <= hi:
			mid = (lo + hi) // 2
			if check(mid):
				ans = mid
				lo = mid + 1
			else:
				hi = mid - 1
		return ans