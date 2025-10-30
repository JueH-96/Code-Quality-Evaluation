import sys
from collections import deque

mod = 998244353

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	out_lines = []
	for _ in range(t):
		N = int(data[index]); index += 1
		p_list = list(map(int, data[index:index+N]))
		index += N
		a_list = list(map(int, data[index:index+N]))
		index += N
		
		children = [[] for _ in range(N+1)]
		for i in range(1, N+1):
			parent = p_list[i-1]
			children[parent].append(i)
		
		depth = [-1] * (N+1)
		q = deque([0])
		depth[0] = 0
		while q:
			u = q.popleft()
			for v in children[u]:
				depth[v] = depth[u] + 1
				q.append(v)
		
		val = [0] * (N+1)
		for i in range(1, N+1):
			val[i] = a_list[i-1]
		
		stack = [0]
		order = []
		while stack:
			u = stack.pop()
			order.append(u)
			for v in children[u]:
				stack.append(v)
				
		S = [0] * (N+1)
		for i in range(len(order)-1, -1, -1):
			u = order[i]
			S[u] = val[u]
			for v in children[u]:
				S[u] += S[v]
				
		total_ops = 0
		for v in range(1, N+1):
			total_ops += a_list[v-1] * depth[v]
			
		for u in range(0, N+1):
			if children[u]:
				lst = [S[child] for child in children[u]]
				lst.sort()
				for idx, s_val in enumerate(lst):
					total_ops += s_val * idx
					
		A = S[0]
		invA = pow(A, mod-2, mod)
		ans = total_ops % mod * invA % mod
		out_lines.append(str(ans))
		
	sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
	main()