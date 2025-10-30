import sys
sys.setrecursionlimit(300000)

class Solution:
	def findAnswer(self, parent: List[int], s: str) -> List[bool]:
		n = len(parent)
		mod1 = 10**9 + 7
		mod2 = 10**9 + 9
		base1 = 131
		base2 = 137
		
		power1 = [1] * (n + 1)
		power2 = [1] * (n + 1)
		for i in range(1, n + 1):
			power1[i] = (power1[i - 1] * base1) % mod1
			power2[i] = (power2[i - 1] * base2) % mod2
		
		children = [[] for _ in range(n)]
		for i in range(1, n):
			p = parent[i]
			children[p].append(i)
		for i in range(n):
			children[i].sort()
		
		size = [0] * n
		F1 = [0] * n
		F2 = [0] * n
		R1 = [0] * n
		R2 = [0] * n
		ans = [False] * n
		
		next_child = [0] * n
		stack = [0]
		
		while stack:
			node = stack[-1]
			if next_child[node] < len(children[node]):
				child = children[node][next_child[node]]
				next_child[node] += 1
				stack.append(child)
			else:
				stack.pop()
				size[node] = 1
				for child in children[node]:
					size[node] += size[child]
				
				F1_val = 0
				F2_val = 0
				for child in children[node]:
					F1_val = (F1_val * power1[size[child]] + F1[child]) % mod1
					F2_val = (F2_val * power2[size[child]] + F2[child]) % mod2
				char_val = ord(s[node]) - ord('a') + 1
				F1_val = (F1_val * base1 + char_val) % mod1
				F2_val = (F2_val * base2 + char_val) % mod2
				F1[node] = F1_val
				F2[node] = F2_val
				
				total_len = size[node] - 1
				R1_val = 0
				R2_val = 0
				for child in reversed(children[node]):
					R1_val = (R1_val * power1[size[child]] + R1[child]) % mod1
					R2_val = (R2_val * power2[size[child]] + R2[child]) % mod2
				R1_val = (char_val * power1[total_len] + R1_val) % mod1
				R2_val = (char_val * power2[total_len] + R2_val) % mod2
				R1[node] = R1_val
				R2[node] = R2_val
				
				if F1_val == R1_val and F2_val == R2_val:
					ans[node] = True
				else:
					ans[node] = False
		
		return ans