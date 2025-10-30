import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n)]
	index = 2
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		w = int(data[index+2])
		index += 3
		u -= 1
		v -= 1
		graph[u].append((v, w))
		graph[v].append((u, w))
	
	ans = None
	stack = [(0, 1, 0)]
	
	while stack:
		u, visited, xor_val = stack.pop()
		if u == n-1:
			if ans is None or xor_val < ans:
				ans = xor_val
			continue
		for v, w in graph[u]:
			if visited & (1 << v):
				continue
			new_visited = visited | (1 << v)
			new_xor = xor_val ^ w
			stack.append((v, new_visited, new_xor))
	
	print(ans)

if __name__ == '__main__':
	main()