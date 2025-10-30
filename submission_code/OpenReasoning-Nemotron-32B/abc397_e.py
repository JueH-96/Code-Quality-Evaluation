import sys
from collections import deque, defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	it = iter(data)
	N = int(next(it)); K = int(next(it))
	total = N * K
	graph = [[] for _ in range(total+1)]
	for i in range(total-1):
		u = int(next(it)); v = int(next(it))
		graph[u].append(v)
		graph[v].append(u)
	
	if K == 1:
		print("Yes")
		return
		
	parent = [0] * (total+1)
	children = [[] for _ in range(total+1)]
	order = []
	q = deque([1])
	parent[1] = 0
	while q:
		u = q.popleft()
		order.append(u)
		for v in graph[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			children[u].append(v)
			q.append(v)
			
	res = [0] * (total+1)
	
	for u in order[::-1]:
		seg = []
		fail = False
		for v in children[u]:
			if res[v] == -1:
				res[u] = -1
				fail = True
				break
			if res[v] != 0:
				seg.append(res[v])
				
		if fail:
			continue
			
		freq = defaultdict(int)
		for x in seg:
			freq[x] += 1
			
		used_u = False
		found_pair = False
		keys = list(freq.keys())
		for a in keys:
			if freq[a] == 0:
				continue
			b = K - 1 - a
			if b < a:
				continue
			if a == b:
				if freq[a] >= 2:
					freq[a] -= 2
					if freq[a] == 0:
						del freq[a]
					used_u = True
					found_pair = True
					break
			else:
				if b in freq and freq[b] > 0:
					freq[a] -= 1
					freq[b] -= 1
					if freq[a] == 0:
						del freq[a]
					if freq[b] == 0:
						del freq[b]
					used_u = True
					found_pair = True
					break
					
		if not found_pair:
			if K-1 in freq and freq[K-1] > 0:
				freq[K-1] -= 1
				if freq[K-1] == 0:
					del freq[K-1]
				used_u = True
				
		remaining = []
		for key, cnt in freq.items():
			for i in range(cnt):
				remaining.append(key)
				
		if len(remaining) > 1:
			res[u] = -1
		else:
			if len(remaining) == 1:
				if used_u:
					res[u] = -1
				else:
					res[u] = remaining[0] + 1
			else:
				if used_u:
					res[u] = 0
				else:
					res[u] = 1
					
	if res[1] == -1:
		print("No")
	elif res[1] == 0 or res[1] == K:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()