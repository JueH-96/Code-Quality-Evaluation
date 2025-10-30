class Solution:
	def furthestDistanceFromOrigin(self, moves: str) -> int:
		count_R = moves.count('R')
		count_L = moves.count('L')
		count_wild = moves.count('_')
		base = count_R - count_L
		return max(abs(base - count_wild), abs(base + count_wild))