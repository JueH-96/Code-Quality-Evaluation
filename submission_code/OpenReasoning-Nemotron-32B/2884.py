from typing import List

class Solution:
	def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
		if not forbidden:
			return len(word)
		
		max_forbidden_len = max(len(f) for f in forbidden)
		forbidden_set = set(forbidden)
		
		left = 0
		max_len_so_far = 0
		
		for right in range(len(word)):
			current_left = left
			window_length = right - left + 1
			check_len = min(max_forbidden_len, window_length)
			for L in range(1, check_len + 1):
				start_index = right - L + 1
				substr = word[start_index:right+1]
				if substr in forbidden_set:
					current_left = max(current_left, start_index + 1)
			if current_left > left:
				left = current_left
			current_length = right - left + 1
			if current_length > max_len_so_far:
				max_len_so_far = current_length
		
		return max_len_so_far