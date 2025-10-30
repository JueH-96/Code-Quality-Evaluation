import heapq
import sys

class SegmentTree:
	def __init__(self, size):
		self.n = size
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.data = [ (10**18, -1) ] * (2 * self.size)
	
	def update(self, index, value):
		if index < 1 or index > self.n:
			return
		leaf = index - 1
		if leaf < 0 or leaf >= self.n:
			return
		leaf += self.size
		self.data[leaf] = value
		while leaf > 1:
			leaf //= 2
			left_val = self.data[2*leaf]
			right_val = self.data[2*leaf+1]
			if left_val[0] < right_val[0]:
				self.data[leaf] = left_val
			else:
				self.data[leaf] = right_val
				
	def query(self, l, r):
		if l > r:
			return None
		l0 = l - 1
		r0 = r - 1
		if l0 < 0:
			l0 = 0
		if r0 >= self.n:
			r0 = self.n - 1
		if l0 > r0:
			return None
		l0 += self.size
		r0 += self.size
		res = (10**18, -1)
		while l0 <= r0:
			if l0 % 2 == 1:
				if self.data[l0][0] < res[0]:
					res = self.data[l0]
				l0 += 1
			if r0 % 2 == 0:
				if self.data[r0][0] < res[0]:
					res = self.data[r0]
				r0 -= 1
			l0 //= 2
			r0 //= 2
		if res[0] == 10**18:
			return None
		return res

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	weights = [int(next(it)) for _ in range(n)]
	intervals = []
	for i in range(n):
		L = int(next(it))
		R = int(next(it))
		intervals.append((L, R, weights[i], i))
	
	intervals_sorted = sorted(intervals, key=lambda x: x[0])
	size = 2 * n
	left_tree = SegmentTree(size)
	right_tree = SegmentTree(size)
	
	left_edges = []
	right_edges = []
	
	for i in range(n):
		L_i, R_i, w_i, global_i = intervals_sorted[i]
		res = left_tree.query(1, L_i - 1)
		if res is not None:
			min_weight, sorted_idx = res
			candidate_global_i = intervals_sorted[sorted_idx][3]
			left_edges.append((global_i, candidate_global_i))
		left_tree.update(R_i, (w_i, i))
	
	for i in range(n-1, -1, -1):
		L_i, R_i, w_i, global_i = intervals_sorted[i]
		res = right_tree.query(R_i + 1, 2 * n)
		if res is not None:
			min_weight, sorted_idx = res
			candidate_global_i = intervals_sorted[sorted_idx][3]
			right_edges.append((global_i, candidate_global_i))
		right_tree.update(L_i, (w_i, i))
	
	graph = [[] for _ in range(n)]
	for u, v in left_edges + right_edges:
		graph[u].append(v)
		graph[v].append(u)
	
	q = int(next(it))
	out_lines = []
	for _ in range(q):
		s = int(next(it)) - 1
		t = int(next(it)) - 1
		if s == t:
			out_lines.append(str(weights[s]))
			continue
		INF = 10**18
		dist = [INF] * n
		dist[s] = weights[s]
		heap = [(dist[s], s)]
		while heap:
			d, u = heapq.heappop(heap)
			if d != dist[u]:
				continue
			if u == t:
				break
			for v in graph[u]:
				new_dist = d + weights[v]
				if new_dist < dist[v]:
					dist[v] = new_dist
					heapq.heappush(heap, (new_dist, v))
		if dist[t] == INF:
			out_lines.append("-1")
		else:
			out_lines.append(str(dist[t]))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()