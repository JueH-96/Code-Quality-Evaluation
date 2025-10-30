class Solution:
	def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
		current_len = 0
		max_len = 0
		last_parity = None
		
		for num in nums:
			if num > threshold:
				current_len = 0
				last_parity = None
			else:
				if last_parity is None:
					if num % 2 == 0:
						current_len = 1
						last_parity = 0
					else:
						current_len = 0
						last_parity = None
				else:
					if num % 2 != last_parity:
						current_len += 1
						last_parity = num % 2
					else:
						if num % 2 == 0:
							current_len = 1
							last_parity = 0
						else:
							current_len = 0
							last_parity = None
				if current_len > max_len:
					max_len = current_len
		return max_len