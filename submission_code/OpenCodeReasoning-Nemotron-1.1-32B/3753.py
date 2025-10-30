class Solution:
	def maxDifference(self, s: str) -> int:
		freq = {}
		for char in s:
			freq[char] = freq.get(char, 0) + 1
		
		evens = []
		odds = []
		for count in freq.values():
			if count % 2 == 0:
				evens.append(count)
			else:
				odds.append(count)
				
		return max(odds) - min(evens)