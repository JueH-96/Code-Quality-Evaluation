import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	index = 1
	for _ in range(n-1):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	best = 0
	for center in range(1, n+1):
		deg = len(graph[center])
		if deg < 2:
			continue
		if deg > 100:
			candidate = 1 + deg
			if candidate > best:
				best = candidate
			continue
		
		neighbor_set = set(graph[center])
		branches = []
		for u in graph[center]:
			count = 0
			for v in graph[u]:
				if v == center:
					continue
				if v in neighbor_set:
					continue
				count += 1
			branches.append((count, u))
		
		branches.sort(key=lambda x: x[0], reverse=True)
		d = len(branches)
		seen = set()
		total_distinct = 0
		for k in range(1, d+1):
			count_val, u = branches[k-1]
			for v in graph[u]:
				if v == center:
					continue
				if v in neighbor_set:
					continue
				if v not in seen:
					seen.add(v)
					total_distinct += 1
			y = min(branches[k-1][0], total_distinct // k)
			candidate = 1 + k * (y + 1)
			if candidate > best:
				best = candidate
				
	print(n - best)

if __name__ == '__main__':
	main()