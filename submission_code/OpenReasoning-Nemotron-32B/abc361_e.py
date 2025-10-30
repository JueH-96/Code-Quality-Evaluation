from collections import deque
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	total_sum = 0
	index = 1
	for _ in range(n-1):
		a = int(data[index])
		b = int(data[index+1])
		c = int(data[index+2])
		index += 3
		graph[a].append((b, c))
		graph[b].append((a, c))
		total_sum += c

	dist1 = [-1] * (n+1)
	q = deque([1])
	dist1[1] = 0
	while q:
		current = q.popleft()
		for neighbor, weight in graph[current]:
			if dist1[neighbor] == -1:
				dist1[neighbor] = dist1[current] + weight
				q.append(neighbor)
	
	u = 1
	for i in range(1, n+1):
		if dist1[i] > dist1[u]:
			u = i

	dist2 = [-1] * (n+1)
	q.append(u)
	dist2[u] = 0
	while q:
		current = q.popleft()
		for neighbor, weight in graph[current]:
			if dist2[neighbor] == -1:
				dist2[neighbor] = dist2[current] + weight
				q.append(neighbor)
	
	diameter = max(dist2[1:])
	ans = 2 * total_sum - diameter
	print(ans)

if __name__ == "__main__":
	main()