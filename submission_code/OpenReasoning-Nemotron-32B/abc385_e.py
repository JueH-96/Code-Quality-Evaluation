import sys
import bisect

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	deg = [0] * (n+1)
	
	idx = 1
	for _ in range(n-1):
		u = int(data[idx])
		v = int(data[idx+1])
		idx += 2
		graph[u].append(v)
		graph[v].append(u)
		deg[u] += 1
		deg[v] += 1
		
	best_kept = 0
	
	for center in range(1, n+1):
		neighbors = graph[center]
		L = []
		for m in neighbors:
			d_val = deg[m] - 1
			if d_val >= 1:
				L.append(d_val)
				
		if not L:
			continue
			
		distinct_vals = {1}
		for d_val in L:
			distinct_vals.add(d_val)
			if d_val - 1 >= 1:
				distinct_vals.add(d_val - 1)
				
		L.sort()
		best_center_val = 0
		for k in distinct_vals:
			pos = bisect.bisect_left(L, k)
			count = len(L) - pos
			total_kept = 1 + count * (k + 1)
			if total_kept > best_center_val:
				best_center_val = total_kept
				
		if best_center_val > best_kept:
			best_kept = best_center_val
			
	ans = n - best_kept
	print(ans)

if __name__ == '__main__':
	main()