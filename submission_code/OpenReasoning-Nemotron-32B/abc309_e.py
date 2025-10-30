import collections
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	parent_list = list(map(int, data[2:2+n-1]))
	
	children = [[] for _ in range(n+1)]
	for i in range(n-1):
		p = parent_list[i]
		child_node = i+2
		children[p].append(child_node)
	
	best = [-10**18] * (n+1)
	index = 2 + n - 1
	for i in range(m):
		x = int(data[index])
		y = int(data[index+1])
		index += 2
		if y > best[x]:
			best[x] = y
			
	depth = [0] * (n+1)
	H = [-10**18] * (n+1)
	
	q = collections.deque()
	q.append(1)
	depth[1] = 0
	H[1] = best[1]
	
	while q:
		u = q.popleft()
		for v in children[u]:
			depth[v] = depth[u] + 1
			H[v] = max(H[u], best[v] + depth[v])
			q.append(v)
			
	count = 0
	for i in range(1, n+1):
		if H[i] >= depth[i]:
			count += 1
			
	print(count)

if __name__ == "__main__":
	main()