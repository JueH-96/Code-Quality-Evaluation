class Solution:
	def maxSubstringLength(self, s: str, k: int) -> bool:
		n = len(s)
		if k == 0:
			return True
		
		first_occurrence = {}
		last_occurrence = {}
		for i, char in enumerate(s):
			if char not in first_occurrence:
				first_occurrence[char] = i
			last_occurrence[char] = i
		
		i = 0
		count = 0
		while i < n and count < k:
			current_min_start = first_occurrence[s[i]]
			current_max_end = last_occurrence[s[i]]
			j = i
			found = False
			while j < n:
				if current_min_start < i:
					break
				if current_max_end <= j:
					if i == 0 and j == n - 1:
						break
					else:
						found = True
						break
				j += 1
				if j < n:
					c = s[j]
					current_min_start = min(current_min_start, first_occurrence[c])
					current_max_end = max(current_max_end, last_occurrence[c])
			
			if found:
				count += 1
				i = j + 1
			else:
				i += 1
		
		return count >= k