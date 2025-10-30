class Solution:
	def maxActiveSectionsAfterTrade(self, s: str) -> int:
		base = s.count('1')
		t = "1" + s + "1"
		groups = []
		current_char = t[0]
		count = 1
		for i in range(1, len(t)):
			if t[i] == current_char:
				count += 1
			else:
				groups.append((current_char, count))
				current_char = t[i]
				count = 1
		groups.append((current_char, count))
		
		n_groups = len(groups)
		if n_groups < 3:
			return base
		
		best_candidate = base
		for i in range(2, n_groups - 1, 2):
			left_zeros = groups[i-1][1]
			right_zeros = groups[i+1][1]
			gain = left_zeros + right_zeros
			candidate = base + gain
			if candidate > best_candidate:
				best_candidate = candidate
				
		return best_candidate