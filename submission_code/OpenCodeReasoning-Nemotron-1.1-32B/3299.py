from collections import Counter
from typing import List

class Solution:
	def maximumLength(self, nums: List[int]) -> int:
		freq = Counter(nums)
		ans = 0
		
		if 1 in freq:
			count1 = freq[1]
			if count1 % 2 == 0:
				count1 -= 1
			ans = count1
		
		for x in freq:
			if x == 1:
				continue
				
			powers = []
			current = x
			while current <= 10**9:
				powers.append(current)
				next_val = current * current
				if next_val > 10**9:
					break
				current = next_val
				
			m = len(powers)
			if m == 0:
				candidate = 0
			else:
				candidate = 1
				for t in range(1, m):
					if freq[powers[t-1]] < 2:
						break
					if freq[powers[t]] < 1:
						break
					candidate = 2 * t + 1
					
			if candidate > ans:
				ans = candidate
				
		return ans