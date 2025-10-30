from typing import List

class Solution:
	def permute(self, n: int, k: int) -> List[int]:
		max_n = 100
		fact = [1] * (max_n + 1)
		for i in range(1, max_n + 1):
			fact[i] = fact[i - 1] * i
		
		odd_count = (n + 1) // 2
		even_count = n // 2
		
		block_size = fact[even_count] * fact[odd_count - 1]
		
		if n % 2 == 0:
			total_blocks = n
		else:
			total_blocks = odd_count
		
		total_permutations = total_blocks * block_size
		
		if k > total_permutations:
			return []
		
		odds = [i for i in range(1, n + 1) if i % 2 == 1]
		evens = [i for i in range(1, n + 1) if i % 2 == 0]
		
		if n % 2 == 0:
			first_elements = list(range(1, n + 1))
		else:
			first_elements = odds
		
		block_index = (k - 1) // block_size
		new_k = (k - 1) % block_size + 1
		
		first_element = first_elements[block_index]
		
		if first_element % 2 == 1:
			new_odds = odds.copy()
			new_odds.remove(first_element)
			new_evens = evens.copy()
			next_parity = 'even'
		else:
			new_evens = evens.copy()
			new_evens.remove(first_element)
			new_odds = odds.copy()
			next_parity = 'odd'
		
		def build_perm(available_odds, available_evens, start_parity, k):
			if not available_odds and not available_evens:
				return []
			if start_parity == 'odd':
				count_per_choice = fact[len(available_evens)] * fact[len(available_odds) - 1]
				index = (k - 1) // count_per_choice
				chosen = available_odds[index]
				new_odds = available_odds[:index] + available_odds[index + 1:]
				new_evens = available_evens
				new_k = (k - 1) % count_per_choice + 1
				return [chosen] + build_perm(new_odds, new_evens, 'even', new_k)
			else:
				count_per_choice = fact[len(available_odds)] * fact[len(available_evens) - 1]
				index = (k - 1) // count_per_choice
				chosen = available_evens[index]
				new_evens = available_evens[:index] + available_evens[index + 1:]
				new_odds = available_odds
				new_k = (k - 1) % count_per_choice + 1
				return [chosen] + build_perm(new_odds, new_evens, 'odd', new_k)
		
		rest = build_perm(new_odds, new_evens, next_parity, new_k)
		return [first_element] + rest