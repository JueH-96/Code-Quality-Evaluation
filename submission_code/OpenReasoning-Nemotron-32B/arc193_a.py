import heapq
import bisect
import sys

sys.setrecursionlimit(300000)

def update_tree(tree, idx, l, r, pos, val):
	if l == r:
		if val < tree[idx]:
			tree[idx] = val
		return
	mid = (l + r) // 2
	if pos <= mid:
		update_tree(tree, idx*2, l, mid, pos, val)
	else:
		update_tree(tree, idx*2+1, mid+1, r, pos, val)
	tree[idx] = min(tree[idx*2], tree[idx*2+1])

def query_tree(tree, idx, l, r, ql, qr):
	if ql > qr:
		return 10**18
	if ql <= l and r <= qr:
		return tree[idx]
	mid = (l + r) // 2
	res = 10**18
	if ql <= mid:
		res = min(res, query_tree(tree, idx*2, l, mid, ql, qr))
	if qr > mid:
		res = min(res, query_tree(tree, idx*2+1, mid+1, r, ql, qr))
	return res

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it))
	weights = [int(next(it)) for _ in range(n)]
	intervals = []
	for _ in range(n):
		L_i = int(next(it))
		R_i = int(next(it))
		intervals.append((L_i, R_i))
	
	q = int(next(it))
	queries = []
	for _ in range(q):
		s = int(next(it)) - 1
		t = int(next(it)) - 1
		queries.append((s, t))
	
	if n <= 2000:
		graph = [[] for _ in range(n)]
		for i in range(n):
			for j in range(i+1, n):
				L_i, R_i = intervals[i]
				L_j, R_j = intervals[j]
				if R_i < L_j or R_j < L_i:
					graph[i].append(j)
					graph[j].append(i)
		
		for s, t in queries:
			if s == t:
				print(0)
				continue
			dist = [10**18] * n
			dist[s] = weights[s]
			pq = [(weights[s], s)]
			found = False
			while pq:
				d, i = heapq.heappop(pq)
				if d != dist[i]:
					continue
				if i == t:
					print(d)
					found = True
					break
				for j in graph[i]:
					new_dist = d + weights[j]
					if new_dist < dist[j]:
						dist[j] = new_dist
						heapq.heappush(pq, (new_dist, j))
			if not found:
				print(-1)
	else:
		size = 2 * n
		for s, t in queries:
			by_r = [[] for _ in range(size+2)]
			by_l = [[] for _ in range(size+2)]
			for i in range(n):
				L_i, R_i = intervals[i]
				by_r[R_i].append(i)
				by_l[L_i].append(i)
			
			dist = [10**18] * n
			dist[s] = weights[s]
			pq = [(weights[s], s)]
			
			active_r_list = []
			active_l_list = []
			for coord in range(size+2):
				if by_r[coord]:
					active_r_list.append(coord)
				if by_l[coord]:
					active_l_list.append(coord)
			active_r_list.sort()
			active_l_list.sort()
			
			tree0 = [10**18] * (4 * (size+1))
			tree1 = [10**18] * (4 * (size+1))
			
			while pq:
				d, i = heapq.heappop(pq)
				if d != dist[i]:
					continue
				L_i, R_i = intervals[i]
				update_tree(tree0, 1, 0, size, R_i, d)
				update_tree(tree1, 1, 0, size, L_i, d)
				
				candidate = query_tree(tree0, 1, 0, size, 0, L_i-1)
				if candidate < 10**18:
					idx = bisect.bisect_left(active_r_list, L_i)
					to_remove = active_r_list[:idx]
					active_r_list = active_r_list[idx:]
					for r_val in to_remove:
						for j in by_r[r_val]:
							new_dist = candidate + weights[j]
							if new_dist < dist[j]:
								dist[j] = new_dist
								heapq.heappush(pq, (new_dist, j))
				
				candidate = query_tree(tree1, 1, 0, size, R_i+1, size)
				if candidate < 10**18:
					idx = bisect.bisect_left(active_l_list, R_i+1)
					to_remove = active_l_list[idx:]
					active_l_list = active_l_list[:idx]
					for l_val in to_remove:
						for j in by_l[l_val]:
							new_dist = candidate + weights[j]
							if new_dist < dist[j]:
								dist[j] = new_dist
								heapq.heappush(pq, (new_dist, j))
			
			if dist[t] == 10**18:
				print(-1)
			else:
				print(dist[t])

if __name__ == "__main__":
	main()