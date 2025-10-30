import itertools
from math import factorial

class Solution:
	def countGoodIntegers(self, n: int, k: int) -> int:
		if n == 0:
			return 0
		
		fact = [1] * (n + 1)
		for i in range(1, n + 1):
			fact[i] = fact[i - 1] * i
		
		total_ans = 0
		for digits in itertools.combinations_with_replacement(range(10), n):
			count = [0] * 10
			for d in digits:
				count[d] += 1
			
			if n % 2 == 0:
				valid = True
				for c in count:
					if c % 2 != 0:
						valid = False
						break
				if not valid:
					continue
				if all(c == 0 for c in count[1:]):
					continue
				
				sym_digits = []
				for d in range(10):
					sym_digits.extend([d] * (count[d] // 2))
				L = n // 2
				found = False
				for perm in itertools.permutations(sym_digits):
					if perm[0] == 0:
						continue
					value = 0
					for i in range(L):
						exponent_left = n - 1 - i
						exponent_right = i
						weight = (10 ** exponent_left) + (10 ** exponent_right)
						value += perm[i] * weight
					if value % k == 0:
						found = True
						break
				if found:
					total_perm = fact[n]
					for c in count:
						total_perm //= fact[c]
					if count[0] > 0:
						invalid_perm = fact[n - 1]
						for d in range(10):
							c_val = count[d]
							if d == 0:
								c_val -= 1
							invalid_perm //= fact[c_val]
						total_valid = total_perm - invalid_perm
					else:
						total_valid = total_perm
					total_ans += total_valid
			else:
				odd_count = 0
				center_candidates = []
				for d in range(10):
					if count[d] % 2 == 1:
						odd_count += 1
						center_candidates.append(d)
				if odd_count != 1:
					continue
				if all(c == 0 for c in count[1:]):
					continue
				found_center = False
				for center in center_candidates:
					sym_digits = []
					for d in range(10):
						if d == center:
							sym_digits.extend([d] * ((count[d] - 1) // 2))
						else:
							sym_digits.extend([d] * (count[d] // 2))
					L = (n - 1) // 2
					if L == 0:
						if center % k == 0:
							found_center = True
							break
						else:
							continue
					found_perm = False
					for perm in itertools.permutations(sym_digits):
						if perm[0] == 0:
							continue
						value_sym = 0
						for i in range(L):
							exponent_left = n - 1 - i
							exponent_right = i
							weight = (10 ** exponent_left) + (10 ** exponent_right)
							value_sym += perm[i] * weight
						total_value = value_sym + center * (10 ** ((n - 1) // 2))
						if total_value % k == 0:
							found_perm = True
							break
					if found_perm:
						found_center = True
						break
				if found_center:
					total_perm = fact[n]
					for c in count:
						total_perm //= fact[c]
					if count[0] > 0:
						invalid_perm = fact[n - 1]
						for d in range(10):
							c_val = count[d]
							if d == 0:
								c_val -= 1
							invalid_perm //= fact[c_val]
						total_valid = total_perm - invalid_perm
					else:
						total_valid = total_perm
					total_ans += total_valid
		return total_ans