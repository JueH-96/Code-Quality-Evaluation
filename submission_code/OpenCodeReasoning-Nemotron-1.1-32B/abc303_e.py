import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	index = 1
	for _ in range(n-1):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	deg = [0] * (n+1)
	for i in range(1, n+1):
		deg[i] = len(graph[i])
	
	leaves = []
	for i in range(1, n+1):
		if deg[i] == 1:
			leaves.append(i)
	
	L = len(leaves)
	M = (n - L + 2) // 3
	
	C0 = set()
	for leaf in leaves:
		neighbor = graph[leaf][0]
		C0.add(neighbor)
	
	if len(C0) == M:
		centers = C0
	else:
		A = set()
		for center in C0:
			for neighbor in graph[center]:
				A.add(neighbor)
		
		leaves_set = set(leaves)
		S = []
		for u in range(1, n+1):
			if u in C0 or u in leaves_set:
				continue
			if u in A:
				continue
			valid = True
			for v in graph[u]:
				if v in A:
					valid = False
					break
			if valid:
				S.append(u)
		
		num_needed = M - len(C0)
		additional = S[:num_needed]
		centers = set(C0) | set(additional)
	
	levels = sorted(deg[u] for u in centers)
	print(" ".join(map(str, levels)))

if __name__ == "__main__":
	main()