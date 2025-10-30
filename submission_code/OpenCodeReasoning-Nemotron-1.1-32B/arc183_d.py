import sys
import heapq
from collections import defaultdict, deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	to = [[] for _ in range(n)]
	index = 1
	for i in range(n-1):
		a = int(data[index]); b = int(data[index+1]); index += 2
		a -= 1
		b -= 1
		to[a].append(b)
		to[b].append(a)
	
	root = 0
	for i in range(n):
		if len(to[i]) > 1:
			root = i
			break
			
	depth = [-1] * n
	parent = [-1] * n
	branch = [-1] * n
	depth[root] = 0
	q = deque([root])
	while q:
		u = q.popleft()
		for v in to[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			depth[v] = depth[u] + 1
			if u == root:
				branch[v] = v
			else:
				branch[v] = branch[u]
			q.append(v)
	
	deg = [0] * n
	for i in range(n):
		deg[i] = len(to[i])
	
	global_heap = []
	branch_heaps = defaultdict(list)
	branch_max = []
	
	for i in range(n):
		if deg[i] == 1:
			heapq.heappush(global_heap, (-depth[i], i, branch[i]))
			heapq.heappush(branch_heaps[branch[i]], (-depth[i], i))
	
	for b in branch_heaps:
		if branch_heaps[b]:
			d, leaf = branch_heaps[b][0]
			heapq.heappush(branch_max, (d, b))
	
	output = []
	for _ in range(n//2):
		leaf1 = None
		b1 = None
		while global_heap:
			d1, candidate, b_candidate = heapq.heappop(global_heap)
			if deg[candidate] != 1:
				continue
			leaf1 = candidate
			b1 = b_candidate
			break
		if leaf1 is None:
			break
			
		leaf2 = None
		b2 = None
		while branch_max:
			d2, b_candidate = heapq.heappop(branch_max)
			if b_candidate == b1:
				continue
			while branch_heaps[b_candidate]:
				d_top, leaf_top = branch_heaps[b_candidate][0]
				if deg[leaf_top] != 1:
					heapq.heappop(branch_heaps[b_candidate])
				else:
					break
			if not branch_heaps[b_candidate]:
				continue
			d_top, leaf_top = branch_heaps[b_candidate][0]
			if d_top == d2:
				leaf2 = leaf_top
				b2 = b_candidate
				break
				
		if leaf2 is None:
			while global_heap:
				d, candidate, b_candidate = heapq.heappop(global_heap)
				if deg[candidate] != 1:
					continue
				if b_candidate != b1:
					leaf2 = candidate
					b2 = b_candidate
					break
			if leaf2 is None:
				while global_heap:
					d, candidate, b_candidate = heapq.heappop(global_heap)
					if deg[candidate] != 1:
						continue
					if b_candidate == b1:
						leaf2 = candidate
						b2 = b1
						break
				if leaf2 is None:
					break
					
		deg[leaf1] = 0
		deg[leaf2] = 0
		output.append((leaf1, leaf2))
		
		p1 = parent[leaf1]
		if p1 != -1:
			deg[p1] -= 1
			if deg[p1] == 1:
				heapq.heappush(global_heap, (-depth[p1], p1, branch[p1]))
				heapq.heappush(branch_heaps[branch[p1]], (-depth[p1], p1))
				heapq.heappush(branch_max, (-depth[p1], branch[p1]))
				
		p2 = parent[leaf2]
		if p2 != -1:
			deg[p2] -= 1
			if deg[p2] == 1:
				heapq.heappush(global_heap, (-depth[p2], p2, branch[p2]))
				heapq.heappush(branch_heaps[branch[p2]], (-depth[p2], p2))
				heapq.heappush(branch_max, (-depth[p2], branch[p2]))
				
	for (u, v) in output:
		print(f"{u+1} {v+1}")

if __name__ == "__main__":
	main()