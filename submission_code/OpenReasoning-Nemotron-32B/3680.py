import math
from collections import defaultdict

class Solution:
	def countComponents(self, nums: list, threshold: int) -> int:
		n = len(nums)
		if n == 0:
			return 0
		
		parent = list(range(n))
		rank = [0] * n
		
		def find(x):
			root = x
			while parent[root] != root:
				root = parent[root]
			while x != root:
				nxt = parent[x]
				parent[x] = root
				x = nxt
			return root
		
		def union(x, y):
			rx = find(x)
			ry = find(y)
			if rx == ry:
				return
			if rank[rx] < rank[ry]:
				parent[rx] = ry
			elif rank[rx] > rank[ry]:
				parent[ry] = rx
			else:
				parent[ry] = rx
				rank[rx] += 1
		
		num_to_index = {num: idx for idx, num in enumerate(nums)}
		
		max_prime = 31622
		sieve = [True] * (max_prime + 1)
		sieve[0] = sieve[1] = False
		primes = []
		for i in range(2, max_prime + 1):
			if sieve[i]:
				primes.append(i)
				start = i * i
				if start > max_prime:
					continue
				for j in range(start, max_prime + 1, i):
					sieve[j] = False
		
		divisor_to_multiples = defaultdict(list)
		
		for num in nums:
			factors = {}
			temp = num
			for p in primes:
				if p * p > temp:
					break
				if temp % p == 0:
					cnt = 0
					while temp % p == 0:
						cnt += 1
						temp //= p
					factors[p] = cnt
			if temp > 1:
				factors[temp] = factors.get(temp, 0) + 1
			
			divisors = {1}
			for p, exp in factors.items():
				new_divisors = set()
				for d in divisors:
					power = 1
					for _ in range(exp + 1):
						new_d = d * power
						if new_d > threshold:
							break
						new_divisors.add(new_d)
						if _ < exp:
							power *= p
				divisors = new_divisors
			
			for d in divisors:
				divisor_to_multiples[d].append(num)
		
		for d, multiples_list in divisor_to_multiples.items():
			if len(multiples_list) < 2:
				continue
			L = sorted(multiples_list)
			T_val = threshold * d
			j = len(L) - 1
			for i in range(len(L)):
				while j > i and L[i] * L[j] > T_val:
					j -= 1
				if j <= i:
					break
				idx_i = num_to_index[L[i]]
				idx_j = num_to_index[L[j]]
				union(idx_i, idx_j)
				if i + 1 < len(L) and i + 1 <= j:
					if L[i] * L[i + 1] <= T_val:
						idx_i1 = num_to_index[L[i + 1]]
						union(idx_i, idx_i1)
		
		components = set()
		for i in range(n):
			components.add(find(i))
		return len(components)