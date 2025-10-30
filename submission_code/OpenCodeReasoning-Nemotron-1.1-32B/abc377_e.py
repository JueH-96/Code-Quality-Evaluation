import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	K = int(data[1])
	P = list(map(int, data[2:2+n]))
	
	B = [x-1 for x in P]
	
	visited = [False] * n
	cycles = []
	for i in range(n):
		if not visited[i]:
			cycle = []
			cur = i
			while not visited[cur]:
				visited[cur] = True
				cycle.append(cur)
				cur = B[cur]
			cycles.append(cycle)
			
	res = [0] * n
	for cycle in cycles:
		L = len(cycle)
		m = pow(2, K, L)
		for j in range(L):
			new_index = (j + m) % L
			res[cycle[j]] = cycle[new_index]
			
	ans = [x+1 for x in res]
	print(" ".join(map(str, ans)))

if __name__ == "__main__":
	main()