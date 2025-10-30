import bisect

class SegmentTree:
	def __init__(self, data):
		self.n = len(data)
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.tree = [[] for _ in range(2 * self.size)]
		for i in range(self.n):
			self.tree[self.size + i] = [data[i]]
		for i in range(self.size - 1, 0, -1):
			left = self.tree[2*i]
			right = self.tree[2*i+1]
			merged = []
			i1, i2 = 0, 0
			while i1 < len(left) and i2 < len(right):
				if left[i1] <= right[i2]:
					merged.append(left[i1])
					i1 += 1
				else:
					merged.append(right[i2])
					i2 += 1
			merged.extend(left[i1:])
			merged.extend(right[i2:])
			self.tree[i] = merged

	def query(self, l, r, A, B):
		l0 = l
		r0 = r
		l += self.size
		r += self.size
		while l <= r:
			if l % 2 == 1:
				arr = self.tree[l]
				pos = bisect.bisect_left(arr, A)
				if pos < len(arr) and arr[pos] <= B:
					return True
				l += 1
			if r % 2 == 0:
				arr = self.tree[r]
				pos = bisect.bisect_left(arr, A)
				if pos < len(arr) and arr[pos] <= B:
					return True
				r -= 1
			l //= 2
			r //= 2
		return False

class Solution:
	def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
		n = len(startTime)
		P = [0] * (n+1)
		for i in range(n):
			P[i+1] = P[i] + (endTime[i] - startTime[i])
		
		st_tree = SegmentTree(P)
		
		def check(x):
			events = []
			for i in range(n):
				events.append((startTime[i] - x, 1, i))
				events.append((endTime[i], -1, i))
			events.sort(key=lambda event: (event[0], event[1]))
			
			min_index = n
			max_index = -1
			n_events = len(events)
			for idx in range(n_events):
				L0 = events[idx][0]
				j = idx
				while j < n_events and events[j][0] == L0:
					typ, i = events[j][1], events[j][2]
					if typ == 1:
						if min_index == n:
							min_index = i
							max_index = i
						else:
							if i == max_index + 1:
								max_index = i
					else:
						if i == min_index:
							min_index += 1
							if min_index > max_index:
								min_index = n
								max_index = -1
					j += 1
				
				if min_index > max_index:
					next_L0 = events[idx+1][0] if idx+1 < n_events else eventTime
					if next_L0 - L0 >= x:
						return True
				else:
					m = max_index - min_index + 1
					if m <= k:
						A_val = P[n] - (eventTime - L0 - x)
						B_val = L0
						if A_val <= B_val:
							if st_tree.query(min_index, max_index+1, A_val, B_val):
								return True
			return False
		
		low, high = 0, eventTime
		ans = 0
		while low <= high:
			mid = (low + high) // 2
			if check(mid):
				ans = mid
				low = mid + 1
			else:
				high = mid - 1
		return ans