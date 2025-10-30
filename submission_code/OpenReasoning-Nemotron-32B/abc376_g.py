mod = 998244353
from functools import cmp_to_key
import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index]); index += 1
		p_list = list(map(int, data[index:index+N])); index += N
		a_list = list(map(int, data[index:index+N])); index += N
		
		children = [[] for _ in range(N+1)]
		for i in range(1, N+1):
			parent = p_list[i-1]
			children[parent].append(i)
		
		size = [0] * (N+1)
		A_arr = [0] * (N+1)
		F = [0] * (N+1)
		
		stack = [0]
		order = []
		while stack:
			u = stack.pop()
			order.append(u)
			for v in reversed(children[u]):
				stack.append(v)
				
		for u in order[::-1]:
			if u == 0:
				base = 0
			else:
				base = a_list[u-1]
				
			size[u] = 1
			A_arr[u] = base
			
			for v in children[u]:
				size[u] += size[v]
				A_arr[u] += A_arr[v]
				
			if children[u]:
				def cmp_func(v1, v2):
					s1, a1 = size[v1], A_arr[v1]
					s2, a2 = size[v2], A_arr[v2]
					prod1 = s1 * a2
					prod2 = s2 * a1
					if prod1 < prod2:
						return -1
					elif prod1 > prod2:
						return 1
					else:
						return 0
					
				sorted_children = sorted(children[u], key=cmp_to_key(cmp_func))
				prefix = 0
				total_sum = 0
				for v in sorted_children:
					term = A_arr[v] * ((prefix + F[v]) % mod) % mod
					total_sum = (total_sum + term) % mod
					prefix += size[v]
					
				if u >= 1:
					invA = pow(A_arr[u], mod-2, mod)
					F[u] = (1 + total_sum * invA) % mod
				else:
					invA = pow(A_arr[u], mod-2, mod)
					F[u] = total_sum * invA % mod
			else:
				if u >= 1:
					F[u] = 1
				else:
					F[u] = 0
					
		results.append(str(F[0]))
	
	sys.stdout.write("
".join(results))

if __name__ == "__main__":
	main()