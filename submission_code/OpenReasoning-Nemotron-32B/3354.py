import heapq

class Solution:
	def minimizeStringValue(self, s: str) -> str:
		n = len(s)
		total_fixed_freq = [0] * 26
		q_total = 0
		for char in s:
			if char != '?':
				idx = ord(char) - ord('a')
				total_fixed_freq[idx] += 1
			else:
				q_total += 1
		
		heap = []
		for i in range(26):
			heapq.heappush(heap, (total_fixed_freq[i], i))
		
		for _ in range(q_total):
			cnt, idx = heapq.heappop(heap)
			heapq.heappush(heap, (cnt + 1, idx))
		
		target_freq = [0] * 26
		while heap:
			cnt, idx = heapq.heappop(heap)
			target_freq[idx] = cnt
		
		current_freq = [0] * 26
		remaining_fixed = total_fixed_freq[:]
		res = []
		
		for char in s:
			if char != '?':
				idx = ord(char) - ord('a')
				current_freq[idx] += 1
				remaining_fixed[idx] -= 1
				res.append(char)
			else:
				for c in "abcdefghijklmnopqrstuvwxyz":
					idx = ord(c) - ord('a')
					if target_freq[idx] - current_freq[idx] - remaining_fixed[idx] - 1 >= 0:
						current_freq[idx] += 1
						res.append(c)
						break
		return ''.join(res)