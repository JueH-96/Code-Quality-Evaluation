def simulate_list(arr, k):
	n = len(arr)
	partitions = 0
	start = 0
	while start < n:
		distinct = 0
		freq = [0] * 26
		for j in range(start, n):
			c = arr[j]
			idx = ord(c) - 97
			if freq[idx] == 0:
				distinct += 1
			freq[idx] += 1
			if distinct > k:
				break
		else:
			j = n - 1
		if distinct > k:
			next_start = j
		else:
			next_start = n
		partitions += 1
		start = next_start
	return partitions

class Solution:
	def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
		s_list = list(s)
		base = simulate_list(s_list, k)
		ans = base
		n = len(s)
		for i in range(n):
			original_char = s_list[i]
			for c in "abcdefghijklmnopqrstuvwxyz":
				if c == original_char:
					continue
				s_list[i] = c
				count = simulate_list(s_list, k)
				if count > ans:
					ans = count
			s_list[i] = original_char
		return ans