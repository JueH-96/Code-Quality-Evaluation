class Solution:
	def minimizeStringValue(self, s: str) -> str:
		freq = [0] * 26
		s_list = list(s)
		for i in range(len(s_list)):
			if s_list[i] != '?':
				idx = ord(s_list[i]) - ord('a')
				freq[idx] += 1
			else:
				min_freq = min(freq)
				for c in range(26):
					if freq[c] == min_freq:
						chosen_char = chr(ord('a') + c)
						break
				s_list[i] = chosen_char
				freq[ord(chosen_char) - ord('a')] += 1
		return ''.join(s_list)