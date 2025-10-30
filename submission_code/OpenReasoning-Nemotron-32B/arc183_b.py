import sys
import heapq
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	
	for _ in range(t):
		N = int(data[index])
		K = int(data[index+1])
		index += 2
		A = list(map(int, data[index:index+N]))
		index += N
		B = list(map(int, data[index:index+N]))
		index += N
		
		heaps = defaultdict(list)
		removed = defaultdict(set)
		
		for i, val in enumerate(A):
			heapq.heappush(heaps[val], i)
		
		possible = True
		for i in range(N):
			v = B[i]
			if v not in heaps or not heaps[v]:
				possible = False
				break
				
			h = heaps[v]
			while h and (h[0] in removed[v] or h[0] < i - K):
				x = heapq.heappop(h)
				if x in removed[v]:
					removed[v].discard(x)
					
			if not h:
				possible = False
				break
				
			if h[0] > i + K:
				possible = False
				break
				
			if A[i] != v:
				old_val = A[i]
				removed[old_val].add(i)
				heapq.heappush(h, i)
				
		results.append("Yes" if possible else "No")
		
	for res in results:
		print(res)

if __name__ == "__main__":
	main()