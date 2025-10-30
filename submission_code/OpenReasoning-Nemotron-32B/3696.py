class Solution:
	def countSubstrings(self, s: str) -> int:
		total_count = 0
		n = len(s)
		for d in range(1, 10):
			rem_count = [0] * d
			for j in range(n):
				digit = int(s[j])
				new_rem_count = [0] * d
				r0 = digit % d
				new_rem_count[r0] += 1
				for r in range(d):
					new_r = (r * 10 + digit) % d
					new_rem_count[new_r] += rem_count[r]
				if digit == d:
					total_count += new_rem_count[0]
				rem_count = new_rem_count
		return total_count