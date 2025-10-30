from collections import Counter

class Solution:
	def maximumLength(self, nums: List[int]) -> int:
		freq = Counter(nums)
		best = 0
		for x in freq:
			if x == 1:
				count1 = freq[1]
				if count1 % 2 == 0:
					chain_length = count1 - 1
				else:
					chain_length = count1
				if chain_length > best:
					best = chain_length
			else:
				terms = []
				current = x
				while current <= 10**9:
					terms.append(current)
					next_peak = current * current
					if next_peak > 10**9:
						break
					current = next_peak
				min_freq_so_far = float('inf')
				max_valid_t = -1
				for t in range(len(terms)):
					if t > 0:
						min_freq_so_far = min(min_freq_so_far, freq[terms[t-1]])
						if min_freq_so_far < 2:
							valid = False
						else:
							valid = True
					else:
						valid = True
					if valid:
						if terms[t] not in freq or freq[terms[t]] < 1:
							valid = False
					if valid:
						max_valid_t = t
					else:
						break
				if max_valid_t == -1:
					chain_length = 0
				else:
					chain_length = 2 * max_valid_t + 1
				if chain_length > best:
					best = chain_length
		return best