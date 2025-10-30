import sys
import heapq
from itertools import product

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	intervals = []
	index = 2
	for i in range(m):
		l = int(data[index])
		r = int(data[index+1])
		index += 2
		intervals.append((l, r))
	
	if m <= 10:
		best_cost = 10**18
		best_assignment = None
		for assignment in product([0, 1, 2], repeat=m):
			coverage_list = []
			cost = 0
			for i in range(m):
				op = assignment[i]
				if op == 0:
					continue
				cost += 1
				l, r = intervals[i]
				if op == 1:
					coverage_list.append((l, r))
				else:
					if l > 1:
						coverage_list.append((1, l-1))
					if r < n:
						coverage_list.append((r+1, n))
			if not coverage_list:
				covered = False
			else:
				coverage_list.sort(key=lambda x: x[0])
				merged = []
				start, end = coverage_list[0]
				for i in range(1, len(coverage_list)):
					seg = coverage_list[i]
					if seg[0] <= end + 1:
						end = max(end, seg[1])
					else:
						merged.append((start, end))
						start, end = seg
				merged.append((start, end))
				if merged[0][0] > 1:
					covered = False
				else:
					if merged[0][1] >= n:
						covered = True
					else:
						covered = False
			if covered:
				if cost < best_cost:
					best_cost = cost
					best_assignment = assignment
		if best_cost == 10**18:
			print(-1)
		else:
			print(best_cost)
			print(" ".join(map(str, best_assignment)))
	else:
		cost1, chosen_indices = greedy_interval_covering(intervals, n, m)
		A0 = 1
		B0 = n
		for (l, r) in intervals:
			if l > A0:
				A0 = l
			if r < B0:
				B0 = r
		if A0 <= B0:
			cost2 = 10**18
		else:
			cost2 = m
		
		if cost1 < 10**18 or cost2 < 10**18:
			if cost1 <= cost2:
				assignment = [0] * m
				for idx in chosen_indices:
					assignment[idx] = 1
				print(cost1)
				print(" ".join(map(str, assignment)))
			else:
				assignment = [2] * m
				print(cost2)
				print(" ".join(map(str, assignment)))
		else:
			print(-1)

def greedy_interval_covering(intervals, n, m):
	intv = []
	for i in range(m):
		l, r = intervals[i]
		intv.append((l, r, i))
	intv.sort(key=lambda x: x[0])
	heap = []
	i_index = 0
	current_cover = 0
	chosen_indices = []
	while current_cover < n:
		while i_index < m and intv[i_index][0] <= current_cover + 1:
			l, r, idx = intv[i_index]
			heapq.heappush(heap, (-r, idx))
			i_index += 1
		if not heap:
			return (10**18, [])
		neg_r, idx = heapq.heappop(heap)
		r_val = -neg_r
		if r_val <= current_cover:
			return (10**18, [])
		current_cover = r_val
		chosen_indices.append(idx)
	return (len(chosen_indices), chosen_indices)

if __name__ == '__main__':
	main()