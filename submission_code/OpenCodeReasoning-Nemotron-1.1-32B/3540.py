class Solution:
	def stringHash(self, s: str, k: int) -> str:
		n = len(s)
		total_chunks = n // k
		result_chars = []
		for i in range(total_chunks):
			start_index = i * k
			end_index = start_index + k
			segment = s[start_index:end_index]
			total = 0
			for char in segment:
				total += ord(char) - ord('a')
			hashed_char = total % 26
			result_chars.append(chr(hashed_char + ord('a')))
		return ''.join(result_chars)