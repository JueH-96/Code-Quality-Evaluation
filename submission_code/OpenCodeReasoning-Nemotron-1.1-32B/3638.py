import itertools
from string import ascii_lowercase

class Solution:
	def makeStringGood(self, s: str) -> int:
		n = len(s)
		best = n - 1
		
		for c in range(26):
			char = chr(ord('a') + c)
			count_le = 0
			sum_cost = 0
			for letter in s:
				if letter <= char:
					count_le += 1
					sum_cost += (ord(char) - ord(letter))
			count_gt = n - count_le
			if count_le > 0:
				cost_val = sum_cost + count_gt
			else:
				cost_val = n + 1
			if cost_val < best:
				best = cost_val
		
		sorted_s = sorted(s)
		letters = list(ascii_lowercase)
		
		for d in range(2, 7):
			for S in itertools.combinations(letters, d):
				S_set = set(S)
				next_d = {}
				for letter in letters:
					next_d[letter] = []
					for d_char in sorted(S_set):
						if d_char >= letter:
							next_d[letter].append(d_char)
				for k in range(1, n + 1):
					if d * k - n > best:
						break
					cap = {d_char: k for d_char in S_set}
					cost = 0
					for c_char in sorted_s:
						assigned = False
						for d_char in next_d[c_char]:
							if cap[d_char] > 0:
								cost += (ord(d_char) - ord(c_char))
								cap[d_char] -= 1
								assigned = True
								break
						if not assigned:
							cost += 1
					insertion_cost = sum(cap.values())
					total_cost = cost + insertion_cost
					if total_cost < best:
						best = total_cost
		return best