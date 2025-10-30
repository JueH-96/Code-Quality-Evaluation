import collections
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	parent_list = list(map(int, data[2:2+n-1]))
	
	children = [[] for _ in range(n)]
	for i in range(1, n):
		p = parent_list[i-1] - 1
		children[p].append(i)
	
	depth = [-1] * n
	depth[0] = 0
	q = collections.deque([0])
	while q:
		u = q.popleft()
		for v in children[u]:
			depth[v] = depth[u] + 1
			q.append(v)
	
	max_y = [-10**15] * n
	index = 2 + n - 1
	for _ in range(m):
		x = int(data[index])
		y = int(data[index+1])
		index += 2
		node_index = x - 1
		if y > max_y[node_index]:
			max_y[node_index] = y
			
	dp = [-10**15] * n
	q = collections.deque([0])
	dp[0] = depth[0] + max_y[0]
	covered_count = 1 if dp[0] >= depth[0] else 0
	
	while q:
		u = q.popleft()
		for v in children[u]:
			dp[v] = max(dp[u], depth[v] + max_y[v])
			if dp[v] >= depth[v]:
				covered_count += 1
			q.append(v)
			
	print(covered_count)

if __name__ == '__main__':
	main()