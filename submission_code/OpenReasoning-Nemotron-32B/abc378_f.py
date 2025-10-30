import sys
sys.setrecursionlimit(300000)

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	deg = [0] * (n+1)
	index = 1
	for _ in range(n-1):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
		deg[u] += 1
		deg[v] += 1

	size = [0] * (n+1)
	for i in range(1, n+1):
		if deg[i] == 2:
			size[i] = 1

	ans = 0

	def dfs(u, pre):
		nonlocal ans
		for v in graph[u]:
			if v == pre:
				continue
			dfs(v, u)
			if deg[u] != 2:
				if size[v] > 0:
					if size[u] > 0:
						ans += size[u] * size[v]
					size[u] += size[v]
			else:
				if size[v] > 0:
					ans += size[v]
					
	dfs(1, 0)
	print(ans)

if __name__ == '__main__':
	main()