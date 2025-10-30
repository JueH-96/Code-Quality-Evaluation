from typing import List

class Solution:
	def permute(self, n: int, k: int) -> List[int]:
		o_rem0 = (n + 1) // 2
		e_rem0 = n // 2
		MAX_K = 10**15
		
		dp0 = [[0] * (e_rem0 + 1) for _ in range(o_rem0 + 1)]
		dp1 = [[0] * (e_rem0 + 1) for _ in range(o_rem0 + 1)]
		
		dp0[0][0] = 1
		dp1[0][0] = 1
		
		for o in range(o_rem0 + 1):
			for e in range(e_rem0 + 1):
				if o > 0:
					prev = dp1[o-1][e]
					if prev == 0:
						dp0[o][e] = 0
					else:
						if prev > MAX_K // o:
							dp0[o][e] = MAX_K + 1
						else:
							product = o * prev
							dp0[o][e] = product if product <= MAX_K else MAX_K + 1
				if e > 0:
					prev = dp0[o][e-1]
					if prev == 0:
						dp1[o][e] = 0
					else:
						if prev > MAX_K // e:
							dp1[o][e] = MAX_K + 1
						else:
							product = e * prev
							dp1[o][e] = product if product <= MAX_K else MAX_K + 1
		
		if n % 2 == 0:
			term1 = o_rem0 * dp1[o_rem0-1][e_rem0]
			term1 = term1 if term1 <= MAX_K else MAX_K + 1
			term2 = e_rem0 * dp0[o_rem0][e_rem0-1]
			term2 = term2 if term2 <= MAX_K else MAX_K + 1
			total_count = term1 + term2
			total_count = total_count if total_count <= MAX_K else MAX_K + 1
		else:
			total_count = o_rem0 * dp1[o_rem0-1][e_rem0]
			total_count = total_count if total_count <= MAX_K else MAX_K + 1
		
		if k > total_count:
			return []
		
		available_odds = [i for i in range(1, n+1) if i % 2 == 1]
		available_evens = [i for i in range(1, n+1) if i % 2 == 0]
		o_rem = o_rem0
		e_rem = e_rem0
		last_parity = None
		res = []
		
		for _ in range(n):
			if last_parity is None:
				if n % 2 == 0:
					candidate_list = sorted(available_odds + available_evens)
				else:
					candidate_list = available_odds
			else:
				if last_parity == 1:
					candidate_list = available_evens
				else:
					candidate_list = available_odds
			
			found = False
			for candidate in candidate_list:
				if candidate % 2 == 1:
					count_rest = dp1[o_rem-1][e_rem]
				else:
					count_rest = dp0[o_rem][e_rem-1]
				
				if k <= count_rest:
					res.append(candidate)
					if candidate % 2 == 1:
						available_odds.remove(candidate)
						o_rem -= 1
						last_parity = 1
					else:
						available_evens.remove(candidate)
						e_rem -= 1
						last_parity = 0
					found = True
					break
				else:
					k -= count_rest
			if not found:
				return []
		return res