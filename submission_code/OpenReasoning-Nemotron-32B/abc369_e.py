import sys
import heapq

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it)); m = int(next(it))
	edges = []
	for _ in range(m):
		u = int(next(it)) - 1
		v = int(next(it)) - 1
		t = int(next(it))
		edges.append((u, v, t))
	
	INF = 10**18
	dist = [[INF] * n for _ in range(n)]
	for i in range(n):
		dist[i][i] = 0
		
	for u, v, t in edges:
		if t < dist[u][v]:
			dist[u][v] = t
			dist[v][u] = t
			
	for k in range(n):
		for i in range(n):
			if dist[i][k] == INF:
				continue
			for j in range(n):
				if dist[i][k] + dist[k][j] < dist[i][j]:
					dist[i][j] = dist[i][k] + dist[k][j]
					
	q = int(next(it))
	out_lines = []
	for _ in range(q):
		k = int(next(it))
		bridge_indices = [int(next(it)) - 1 for _ in range(k)]
		edges_req = []
		V0_set = set()
		V0_set.add(0)
		V0_set.add(n-1)
		for idx in bridge_indices:
			u, v, t = edges[idx]
			edges_req.append((u, v, t))
			V0_set.add(u)
			V0_set.add(v)
			
		V0 = sorted(V0_set)
		K0 = len(V0)
		node_to_index = {node: idx for idx, node in enumerate(V0)}
		
		dist_v0 = [[INF] * K0 for _ in range(K0)]
		for i in range(K0):
			for j in range(K0):
				u_node = V0[i]
				v_node = V0[j]
				dist_v0[i][j] = dist[u_node][v_node]
				
		num_masks = 1 << k
		dp = [[INF] * K0 for _ in range(num_masks)]
		heap = []
		start_index = node_to_index[0]
		dp[0][start_index] = 0
		heapq.heappush(heap, (0, 0, start_index))
		
		end_index = node_to_index[n-1]
		ans = INF
		
		while heap:
			cost, mask, v_idx = heapq.heappop(heap)
			if cost != dp[mask][v_idx]:
				continue
			if mask == num_masks - 1 and v_idx == end_index:
				ans = cost
				break
				
			for w_idx in range(K0):
				new_cost = cost + dist_v0[v_idx][w_idx]
				if new_cost < dp[mask][w_idx]:
					dp[mask][w_idx] = new_cost
					heapq.heappush(heap, (new_cost, mask, w_idx))
					
			for j in range(k):
				if mask & (1 << j):
					continue
				new_mask = mask | (1 << j)
				u_edge, v_edge, w_edge = edges_req[j]
				a_idx = node_to_index[u_edge]
				b_idx = node_to_index[v_edge]
				cost1 = cost + dist_v0[v_idx][a_idx] + w_edge
				if cost1 < dp[new_mask][b_idx]:
					dp[new_mask][b_idx] = cost1
					heapq.heappush(heap, (cost1, new_mask, b_idx))
				cost2 = cost + dist_v0[v_idx][b_idx] + w_edge
				if cost2 < dp[new_mask][a_idx]:
					dp[new_mask][a_idx] = cost2
					heapq.heappush(heap, (cost2, new_mask, a_idx))
					
		if ans == INF:
			ans = dp[num_masks-1][end_index]
		out_lines.append(str(ans))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()