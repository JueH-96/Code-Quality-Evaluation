class Solution:
	def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
		n = len(s)
		if n == 0:
			return 0
		
		breaks = [0]
		i = 0
		while i < n:
			freq = [0] * 26
			distinct = 0
			j = i
			while j < n:
				idx = ord(s[j]) - ord('a')
				if freq[idx] == 0:
					if distinct == k:
						break
					distinct += 1
				freq[idx] += 1
				j += 1
			breaks.append(j)
			i = j
		
		m = len(breaks) - 1
		seg_index = [-1] * n
		for seg_id in range(m):
			start = breaks[seg_id]
			end = breaks[seg_id+1]
			for idx in range(start, end):
				seg_index[idx] = seg_id
		
		ans = m
		
		for i in range(n):
			for c in "abcdefghijklmnopqrstuvwxyz":
				if c == s[i]:
					continue
				seg_id = seg_index[i]
				start_seg = breaks[seg_id]
				count_seg = 0
				p = start_seg
				while p < n:
					freq = [0] * 26
					distinct = 0
					q = p
					while q < n:
						if q == i:
							char = c
						else:
							char = s[q]
						idx_char = ord(char) - ord('a')
						if freq[idx_char] == 0:
							if distinct == k:
								break
							distinct += 1
						freq[idx_char] += 1
						q += 1
					count_seg += 1
					p = q
				total_partitions = seg_id + count_seg
				if total_partitions > ans:
					ans = total_partitions
		return ans