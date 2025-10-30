MOD1 = 10**9 + 7
MOD2 = 10**9 + 9
BASE1 = 131
BASE2 = 137

class Solution:
	def findAnswer(self, parent: List[int], s: str) -> List[bool]:
		n = len(parent)
		children = [[] for _ in range(n)]
		for i in range(1, n):
			p = parent[i]
			children[p].append(i)
		
		for i in range(n):
			children[i].sort()
		
		stack = [0]
		visited = [False] * n
		S = []
		start = [0] * n
		end = [0] * n
		
		while stack:
			node = stack.pop()
			if visited[node]:
				S.append(s[node])
				end[node] = len(S) - 1
			else:
				visited[node] = True
				stack.append(node)
				for child in reversed(children[node]):
					stack.append(child)
				start[node] = len(S)
		
		T = S[::-1]
		
		max_power = n
		pow1 = [1] * (max_power + 1)
		pow2 = [1] * (max_power + 1)
		for i in range(1, max_power + 1):
			pow1[i] = (pow1[i-1] * BASE1) % MOD1
			pow2[i] = (pow2[i-1] * BASE2) % MOD2
		
		H1_S = [0] * n
		H2_S = [0] * n
		if n > 0:
			H1_S[0] = ord(S[0]) % MOD1
			H2_S[0] = ord(S[0]) % MOD2
			for i in range(1, n):
				H1_S[i] = (H1_S[i-1] * BASE1 + ord(S[i])) % MOD1
				H2_S[i] = (H2_S[i-1] * BASE2 + ord(S[i])) % MOD2
		
		H1_T = [0] * n
		H2_T = [0] * n
		if n > 0:
			H1_T[0] = ord(T[0]) % MOD1
			H2_T[0] = ord(T[0]) % MOD2
			for i in range(1, n):
				H1_T[i] = (H1_T[i-1] * BASE1 + ord(T[i])) % MOD1
				H2_T[i] = (H2_T[i-1] * BASE2 + ord(T[i])) % MOD2
		
		def get_hash(l, r, H, pow_arr, mod):
			if l > r:
				return 0
			if l == 0:
				return H[r]
			h = (H[r] - H[l-1] * pow_arr[r - l + 1]) % mod
			return h
		
		ans = [False] * n
		for x in range(n):
			l = start[x]
			r = end[x]
			rev_l = n - 1 - r
			rev_r = n - 1 - l
			
			h1_S = get_hash(l, r, H1_S, pow1, MOD1)
			h2_S = get_hash(l, r, H2_S, pow2, MOD2)
			
			h1_T = get_hash(rev_l, rev_r, H1_T, pow1, MOD1)
			h2_T = get_hash(rev_l, rev_r, H2_T, pow2, MOD2)
			
			if (h1_S, h2_S) == (h1_T, h2_T):
				ans[x] = True
			else:
				ans[x] = False
		
		return ans