import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	edges = []
	index = 1
	adj = [[] for _ in range(n+1)]
	for i in range(n-1):
		u = int(data[index]); v = int(data[index+1]); index += 2
		adj[u].append(v)
		adj[v].append(u)
	
	parent = [0] * (n+1)
	order = []
	stack = [1]
	parent[1] = 0
	while stack:
		u = stack.pop()
		order.append(u)
		for v in adj[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			stack.append(v)
	
	NEG = -10**18
	dp0 = [NEG] * (n+1)
	dp1 = [NEG] * (n+1)
	global_ans = NEG

	for u in order[::-1]:
		children = [v for v in adj[u] if v != parent[u]]
		arr = []
		for v in children:
			if dp1[v] >= 0:
				arr.append(dp1[v])
				
		if len(arr) == 0:
			dp0[u] = NEG
			dp1[u] = 1
		else:
			top = [NEG] * 4
			for x in arr:
				if x > top[0]:
					top[3] = top[2]
					top[2] = top[1]
					top[1] = top[0]
					top[0] = x
				elif x > top[1]:
					top[3] = top[2]
					top[2] = top[1]
					top[1] = x
				elif x > top[2]:
					top[3] = top[2]
					top[2] = x
				elif x > top[3]:
					top[3] = x
					
			cand0_1 = NEG
			cand0_4 = NEG
			if top[0] != NEG:
				cand0_1 = 1 + top[0]
			if top[3] != NEG:
				cand0_4 = 1 + top[0] + top[1] + top[2] + top[3]
			dp0[u] = cand0_1 if cand0_1 > cand0_4 else cand0_4
			
			cand1_0 = 1
			cand1_3 = NEG
			if top[2] != NEG:
				cand1_3 = 1 + top[0] + top[1] + top[2]
			dp1[u] = cand1_0 if cand1_0 > cand1_3 else cand1_3
			
		if dp0[u] >= 5:
			if dp0[u] > global_ans:
				global_ans = dp0[u]
				
	if global_ans < 0:
		print(-1)
	else:
		print(global_ans)

if __name__ == "__main__":
	main()