import sys
sys.setrecursionlimit(1000000)

class Solution:
	def maxSubstringLength(self, s: str, k: int) -> bool:
		n = len(s)
		if k == 0:
			return True
		
		first_occ = {}
		last_occ = {}
		for idx, char in enumerate(s):
			if char not in first_occ:
				first_occ[char] = idx
			last_occ[char] = idx
		
		A = [0] * n
		B = [0] * n
		for i, char in enumerate(s):
			A[i] = last_occ[char]
			B[i] = first_occ[char]
		
		log = [0] * (n + 1)
		for i in range(2, n + 1):
			log[i] = log[i // 2] + 1
		num_log = log[n] + 1
		
		st_max = [[0] * n for _ in range(num_log)]
		for i in range(n):
			st_max[0][i] = A[i]
		for i in range(1, num_log):
			step = 1 << (i - 1)
			j = 0
			while j + (1 << i) <= n:
				st_max[i][j] = max(st_max[i - 1][j], st_max[i - 1][j + step])
				j += 1
		
		st_min = [[0] * n for _ in range(num_log)]
		for i in range(n):
			st_min[0][i] = B[i]
		for i in range(1, num_log):
			step = 1 << (i - 1)
			j = 0
			while j + (1 << i) <= n:
				st_min[i][j] = min(st_min[i - 1][j], st_min[i - 1][j + step])
				j += 1
		
		def query_max(l, r):
			if l > r:
				return -10**9
			length = r - l + 1
			k_val = log[length]
			return max(st_max[k_val][l], st_max[k_val][r - (1 << k_val) + 1])
		
		def query_min(l, r):
			if l > r:
				return 10**9
			length = r - l + 1
			k_val = log[length]
			return min(st_min[k_val][l], st_min[k_val][r - (1 << k_val) + 1])
		
		valid = [False] * n
		closure_end = [-1] * n
		for i in range(n):
			if first_occ[s[i]] != i:
				valid[i] = False
				closure_end[i] = -1
			else:
				r = last_occ[s[i]]
				while True:
					M = query_max(i, r)
					if M > r:
						r = M
					else:
						break
				min_first_in_segment = query_min(i, r)
				if min_first_in_segment < i:
					valid[i] = False
					closure_end[i] = -1
				else:
					valid[i] = True
					closure_end[i] = r
		
		memo = {}
		def dfs(i, count):
			if count >= k:
				return True
			if i >= n:
				return False
			key = (i, count)
			if key in memo:
				return memo[key]
			
			if dfs(i + 1, count):
				memo[key] = True
				return True
			
			if valid[i]:
				r = closure_end[i]
				if i == 0 and r == n - 1:
					pass
				else:
					if dfs(r + 1, count + 1):
						memo[key] = True
						return True
			
			memo[key] = False
			return False
		
		return dfs(0, 0)