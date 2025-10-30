class Solution:
	def maxLength(self, nums: List[int]) -> int:
		primes = [2, 3, 5, 7]
		n = len(nums)
		precomputed = []
		for num in nums:
			exp_dict = {}
			for p in primes:
				cnt = 0
				temp = num
				while temp % p == 0:
					cnt += 1
					temp //= p
				exp_dict[p] = cnt
			precomputed.append(exp_dict)
		
		max_len = 0
		for start in range(n):
			min_exp = {p: float('inf') for p in primes}
			max_exp = {p: float('-inf') for p in primes}
			total_exp = {p: 0 for p in primes}
			
			for end in range(start, n):
				exp_dict = precomputed[end]
				for p in primes:
					e = exp_dict[p]
					if e < min_exp[p]:
						min_exp[p] = e
					if e > max_exp[p]:
						max_exp[p] = e
					total_exp[p] += e
				
				valid = True
				for p in primes:
					if total_exp[p] - min_exp[p] != max_exp[p]:
						valid = False
						break
				
				if valid:
					length = end - start + 1
					if length > max_len:
						max_len = length
		
		return max_len