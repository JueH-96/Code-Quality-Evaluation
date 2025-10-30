import math
from typing import List

class Solution:
	_spf_arr = None
	
	def canTraverseAllPairs(self, nums: List[int]) -> bool:
		n = len(nums)
		if n == 1:
			return True
		
		if Solution._spf_arr is None:
			max_val = 100000
			spf_arr = list(range(max_val + 1))
			sqrt_max_val = math.isqrt(max_val)
			for i in range(2, sqrt_max_val + 1):
				if spf_arr[i] == i:
					for j in range(i * i, max_val + 1, i):
						if spf_arr[j] == j:
							spf_arr[j] = i
			Solution._spf_arr = spf_arr
		
		spf_arr = Solution._spf_arr
		parent = list(range(n))
		rank = [0] * n
		
		def find(x: int, parent: List[int]) -> int:
			root = x
			while root != parent[root]:
				root = parent[root]
			while x != root:
				next_node = parent[x]
				parent[x] = root
				x = next_node
			return root
		
		prime_to_index = {}
		
		for i in range(n):
			num = nums[i]
			if num == 1:
				continue
			factors = set()
			temp = num
			while temp > 1:
				p = spf_arr[temp]
				factors.add(p)
				while temp % p == 0:
					temp //= p
			for p in factors:
				if p in prime_to_index:
					j = prime_to_index[p]
					root_i = find(i, parent)
					root_j = find(j, parent)
					if root_i != root_j:
						if rank[root_i] < rank[root_j]:
							parent[root_i] = root_j
						elif rank[root_i] > rank[root_j]:
							parent[root_j] = root_i
						else:
							parent[root_j] = root_i
							rank[root_i] += 1
				else:
					prime_to_index[p] = i
		
		roots = set()
		for i in range(n):
			roots.add(find(i, parent))
		
		return len(roots) == 1