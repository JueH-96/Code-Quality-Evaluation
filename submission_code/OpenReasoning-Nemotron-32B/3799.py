from collections import Counter
from typing import List

class Solution:
	def totalNumbers(self, digits: List[int]) -> int:
		freq = Counter(digits)
		count = 0
		for last in [0, 2, 4, 6, 8]:
			if freq[last] > 0:
				freq[last] -= 1
				for first in freq:
					if first == 0:
						continue
					if freq[first] > 0:
						freq[first] -= 1
						for mid in freq:
							if freq[mid] > 0:
								count += 1
						freq[first] += 1
				freq[last] += 1
		return count