import sys
import heapq

inf = 10**18

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	it = iter(data)
	n = int(next(it)); q = int(next(it))
	ops = []
	for i in range(q):
		L = int(next(it)); R = int(next(it)); C = int(next(it))
		ops.append((L, R, C))
	
	diff_cover = [0] * (n+3)
	for (L, R, C) in ops:
		diff_cover[L] += 1
		if R+1 <= n:
			diff_cover[R+1] -= 1
	
	cover_count = [0] * (n+1)
	for i in range(1, n+1):
		cover_count[i] = diff_cover[i] + (cover_count[i-1] if i-1>=1 else 0)
	
	for i in range(1, n+1):
		if cover_count[i] == 0:
			print(-1)
			return

	diff_gap = [0] * (n+3)
	for (L, R, C) in ops:
		if L <= R-1:
			diff_gap[L] += 1
			if R <= n:
				diff_gap[R] -= 1
	
	if n >= 2:
		gap_cover_count = [0] * (n+1)
		for i in range(1, n):
			gap_cover_count[i] = diff_gap[i] + (gap_cover_count[i-1] if i-1>=1 else 0)
		for i in range(1, n):
			if gap_cover_count[i] == 0:
				print(-1)
				return
	else:
		if n == 1:
			pass
		else:
			print(-1)
			return

	events = [[] for _ in range(n+2)]
	for i, (L, R, C) in enumerate(ops):
		events[L].append((1, C, i))
		events[R+1].append((-1, C, i))
	
	min_cost_arr = [0] * (n+1)
	min_op_arr = [-1] * (n+1)
	active = []
	to_remove = []
	
	for u in range(1, n+1):
		for event in events[u]:
			typ, cost, op_idx = event
			if typ == 1:
				heapq.heappush(active, (cost, op_idx))
			else:
				heapq.heappush(to_remove, (cost, op_idx))
		while active and to_remove and active[0] == to_remove[0]:
			heapq.heappop(active)
			heapq.heappop(to_remove)
		if active:
			cost_val, op_idx = active[0]
			min_cost_arr[u] = cost_val
			min_op_arr[u] = op_idx
		else:
			min_cost_arr[u] = inf
			min_op_arr[u] = -1

	total_cost = sum(min_cost_arr[1:])
	
	parent = list(range(q))
	size = [0] * q
	comp_list = [[] for _ in range(q)]
	for u in range(1, n+1):
		if min_op_arr[u] != -1:
			comp_list[min_op_arr[u]].append(u)
	for i in range(q):
		size[i] = len(comp_list[i])
	
	comp_id = [0] * (n+1)
	for u in range(1, n+1):
		if min_op_arr[u] != -1:
			comp_id[u] = min_op_arr[u]
		else:
			comp_id[u] = -1

	data_arr = [0] * n
	for i in range(1, n+1):
		data_arr[i-1] = comp_id[i]
	
	class SegmentTree:
		def __init__(self, data):
			self.n = len(data)
			self.size = 1
			while self.size < self.n:
				self.size *= 2
			self.min1 = [inf] * (2 * self.size)
			self.min2 = [inf] * (2 * self.size)
			for i in range(self.n):
				self.min1[self.size + i] = data[i]
				self.min2[self.size + i] = inf
			for i in range(self.n, self.size):
				self.min1[self.size + i] = inf
				self.min2[self.size + i] = inf
			for i in range(self.size-1, 0, -1):
				self._combine(i)
				
		def _combine(self, i):
			left_min1 = self.min1[2*i]
			left_min2 = self.min2[2*i]
			right_min1 = self.min1[2*i+1]
			right_min2 = self.min2[2*i+1]
			candidates = [left_min1, left_min2, right_min1, right_min2]
			candidates.sort()
			min1 = candidates[0]
			min2 = inf
			for i in range(1, 4):
				if candidates[i] != min1:
					min2 = candidates[i]
					break
			self.min1[i] = min1
			self.min2[i] = min2
		
		def update(self, index, value):
			i = self.size + index
			self.min1[i] = value
			self.min2[i] = inf
			i //= 2
			while i:
				self._combine(i)
				i //= 2
		
		def query(self, l, r):
			l0 = l
			r0 = r
			l += self.size
			r += self.size
			segments = []
			while l <= r:
				if l % 2 == 1:
					segments.append((self.min1[l], self.min2[l]))
					l += 1
				if r % 2 == 0:
					segments.append((self.min1[r], self.min2[r]))
					r -= 1
				l //= 2
				r //= 2
			min1 = inf
			min2 = inf
			for seg in segments:
				a, b = seg
				if a < min1:
					min2 = min1
					min1 = a
				elif a == min1:
					if b < min2:
						min2 = b
				if a < min2 and a != min1:
					min2 = a
				if b < min2 and b != min1:
					min2 = b
			return (min1, min2)
	
	seg_tree = SegmentTree(data_arr)
	
	sorted_ops = sorted(enumerate(ops), key=lambda x: x[1][2])
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	for idx, (L, R, C) in sorted_ops:
		comp_i = find(idx)
		l_index = L-1
		r_index = R-1
		if l_index > r_index:
			continue
		min1, min2 = seg_tree.query(l_index, r_index)
		candidate_comp = None
		if min1 != inf and min1 != comp_i:
			candidate_comp = min1
		elif min2 != inf:
			candidate_comp = min2
		else:
			continue
		
		comp_j = candidate_comp
		comp_j_rep = find(comp_j)
		if comp_i == comp_j_rep:
			continue
		if size[comp_i] < size[comp_j_rep]:
			comp_i, comp_j_rep = comp_j_rep, comp_i
		parent[comp_j_rep] = comp_i
		size[comp_i] += size[comp_j_rep]
		for u in comp_list[comp_j_rep]:
			comp_id[u] = comp_i
			seg_tree.update(u-1, comp_i)
			comp_list[comp_i].append(u)
		comp_list[comp_j_rep] = []
		total_cost += C

	print(total_cost)

if __name__ == '__main__':
	main()