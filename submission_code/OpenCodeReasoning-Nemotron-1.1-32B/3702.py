from typing import List

class Solution:
	def maxLength(self, nums: List[int]) -> int:
		primes = [2, 3, 5, 7]
		n = len(nums)
		factor_exponents = []
		for num in nums:
			exp_list = []
			for p in primes:
				cnt = 0
				temp = num
				while temp % p == 0:
					cnt += 1
					temp //= p
				exp_list.append(cnt)
			factor_exponents.append(exp_list)
		
		max_len = 0
		for start in range(n):
			for end in range(start, n):
				length = end - start + 1
				valid = True
				for j in range(4):
					exp_list = [factor_exponents[i][j] for i in range(start, end + 1)]
					total = sum(exp_list)
					min_exp = min(exp_list)
					max_exp = max(exp_list)
					if total != min_exp + max_exp:
						valid = False
						break
				if valid:
					if length > max_len:
						max_len = length
		return max_len