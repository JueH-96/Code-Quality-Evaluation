import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	edges = []
	index = 2
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		edges.append((u, v))
	
	W = [0] + list(map(int, data[index:index+n]))
	index += n
	A = [0] + list(map(int, data[index:index+n]))
	index += n

	graph = [[] for _ in range(n+1)]
	for u, v in edges:
		graph[u].append(v)
		graph[v].append(u)
	
	vertices = list(range(1, n+1))
	vertices.sort(key=lambda i: W[i])
	
	f = [0] * (n+1)
	opt_set = [set() for _ in range(n+1)]
	
	for i in vertices:
		neighbors = []
		for j in graph[i]:
			if W[j] < W[i]:
				neighbors.append(j)
				
		if not neighbors:
			f[i] = 1
			opt_set[i] = set()
		else:
			cap_max = W[i] - 1
			if cap_max <= 0:
				f[i] = 1
				opt_set[i] = set()
			else:
				items = []
				for j in neighbors:
					items.append((W[j], f[j], j))
				
				dp_arr = [0] * (cap_max+1)
				for w, val, j in items:
					for cap in range(cap_max, w-1, -1):
						if dp_arr[cap] < dp_arr[cap - w] + val:
							dp_arr[cap] = dp_arr[cap - w] + val
				
				max_val = max(dp_arr) if dp_arr else 0
				f[i] = 1 + max_val
				
				if max_val == 0:
					opt_set[i] = set()
				else:
					cap0 = -1
					for cap in range(cap_max+1):
						if dp_arr[cap] == max_val:
							cap0 = cap
							break
					if cap0 == -1:
						opt_set[i] = set()
					else:
						current_cap = cap0
						chosen = set()
						for idx in range(len(items)-1, -1, -1):
							w, val, j = items[idx]
							if current_cap >= w and dp_arr[current_cap] == dp_arr[current_cap - w] + val:
								chosen.add(j)
								current_cap -= w
						opt_set[i] = chosen
						
	g = A[:]
	vertices.sort(key=lambda i: W[i], reverse=True)
	
	for i in vertices:
		for j in opt_set[i]:
			g[j] += g[i]
			
	ans = sum(g[1:])
	print(ans)

if __name__ == '__main__':
	main()