import itertools

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	k = int(data[2])
	edges = []
	index = 3
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		w = int(data[index+2])
		index += 3
		edges.append((u, v, w))
	
	min_cost = float('inf')

	for comb in itertools.combinations(edges, n-1):
		parent = list(range(n+1))
		total = 0
		valid = True
		for u, v, w in comb:
			ru = u
			while parent[ru] != ru:
				ru = parent[ru]
			rv = v
			while parent[rv] != rv:
				rv = parent[rv]
			if ru == rv:
				valid = False
				break
			parent[rv] = ru
			total += w
		
		if not valid:
			continue
		
		residue = total % k
		if residue < min_cost:
			min_cost = residue
	
	print(min_cost)

if __name__ == '__main__':
	main()