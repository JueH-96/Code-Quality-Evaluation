import re

class Solution:
	def maxActiveSectionsAfterTrade(self, s: str) -> int:
		base = s.count('1')
		t = '1' + s + '1'
		ones_blocks = re.findall(r'1+', t)
		zeros_blocks = re.findall(r'0+', t)
		
		if len(ones_blocks) < 3:
			return base
		
		max_gain = 0
		for i in range(1, len(ones_blocks) - 1):
			gain = len(zeros_blocks[i-1]) + len(zeros_blocks[i])
			if gain > max_gain:
				max_gain = gain
		
		return base + max_gain