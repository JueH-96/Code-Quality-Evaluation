import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	P = list(map(int, data[2:2+n]))
	
	P0 = [x - 1 for x in P]
	
	visited = [False] * n
	cycles = []
	for i in range(n):
		if not visited[i]:
			cycle = []
			cur = i
			while not visited[cur]:
				visited[cur] = True
				cycle.append(cur)
				cur = P0[cur]
			cycles.append(cycle)
	
	res = [0] * n
	
	for cycle in cycles:
		L = len(cycle)
		if L == 1:
			m = 0
		else:
			m = pow(2, k, L)
		for j, elem in enumerate(cycle):
			new_index = (j + m) % L
			res[elem] = cycle[new_index]
	
	res_1_indexed = [x + 1 for x in res]
	print(" ".join(map(str, res_1_indexed)))

if __name__ == '__main__':
	main()