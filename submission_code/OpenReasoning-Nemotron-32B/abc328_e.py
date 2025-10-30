import itertools
import sys

def find(x, parent):
	if parent[x] != x:
		parent[x] = find(parent[x], parent)
	return parent[x]

def union(x, y, parent, rank):
	rx = find(x, parent)
	ry = find(y, parent)
	if rx == ry:
		return
	if rank[rx] < rank[ry]:
		parent[rx] = ry
	elif rank[rx] > rank[ry]:
		parent[ry] = rx
	else:
		parent[ry] = rx
		rank[rx] += 1

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	K_val = int(data[2])
	edges = []
	idx = 3
	for i in range(m):
		u = int(data[idx])
		v = int(data[idx+1])
		w = int(data[idx+2])
		idx += 3
		edges.append((u-1, v-1, w))
	
	min_cost = K_val
	if n == 1:
		print(0)
		return
		
	from itertools import combinations
	for subset in combinations(edges, n-1):
		total = 0
		for edge in subset:
			total += edge[2]
			
		parent = list(range(n))
		rank = [0] * n
		valid = True
		for (u, v, w) in subset:
			ru = find(u, parent)
			rv = find(v, parent)
			if ru == rv:
				valid = False
				break
			union(ru, rv, parent, rank)
			
		if not valid:
			continue
			
		candidate = total % K_val
		if candidate < min_cost:
			min_cost = candidate
			if min_cost == 0:
				break
				
	print(min_cost)

if __name__ == "__main__":
	main()