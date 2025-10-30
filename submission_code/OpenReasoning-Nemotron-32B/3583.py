from typing import List

class Solution:
	def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
		max_val = 50000
		n = len(nums)
		freq = [0] * (max_val + 1)
		for x in nums:
			if x <= max_val:
				freq[x] += 1
		
		mu = [1] * (max_val + 1)
		is_prime = [True] * (max_val + 1)
		primes = []
		is_prime[0] = False
		is_prime[1] = False
		for i in range(2, max_val + 1):
			if is_prime[i]:
				primes.append(i)
				mu[i] = -1
			for p in primes:
				if i * p > max_val:
					break
				is_prime[i * p] = False
				if i % p == 0:
					mu[i * p] = 0
					break
				else:
					mu[i * p] = -mu[i]
		
		count_arr = [0] * (max_val + 1)
		for d in range(1, max_val + 1):
			total = 0
			for j in range(d, max_val + 1, d):
				total += freq[j]
			count_arr[d] = total
		
		f_arr = [0] * (max_val + 1)
		for d in range(1, max_val + 1):
			total = 0
			k = 1
			while d * k <= max_val:
				total += mu[k] * (count_arr[d * k] * (count_arr[d * k] - 1) // 2)
				k += 1
			f_arr[d] = total
		
		prefix = [0] * (max_val + 1)
		for d in range(1, max_val + 1):
			prefix[d] = prefix[d - 1] + f_arr[d]
		
		ans = []
		for q in queries:
			low, high = 1, max_val
			while low < high:
				mid = (low + high) // 2
				if prefix[mid] <= q:
					low = mid + 1
				else:
					high = mid
			ans.append(low)
		
		return ans