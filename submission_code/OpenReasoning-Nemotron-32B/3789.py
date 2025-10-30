import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	conflictingPairs = []
	index = 1
	for i in range(len(conflictingPairs) * 2):
		a = int(data[index]); b = int(data[index+1]); index += 2
		conflictingPairs.append([a, b])
	
	total_subarrays = n * (n + 1) // 2
	INF = 10**9 + 10

	min_b_from_i = [INF] * (n + 2)
	for a, b in conflictingPairs:
		a1 = min(a, b)
		b1 = max(a, b)
		if b1 < min_b_from_i[a1]:
			min_b_from_i[a1] = b1

	min_b_arr = [INF] * (n + 2)
	min_b_arr[n] = min_b_from_i[n]
	for i in range(n - 1, 0, -1):
		min_b_arr[i] = min(min_b_arr[i + 1], min_b_from_i[i])
	
	total_union = 0
	for i in range(1, n + 1):
		if min_b_arr[i] != INF:
			total_union += n - min_b_arr[i] + 1

	count_b = [0] * (n + 2)
	for a, b in conflictingPairs:
		a1 = min(a, b)
		b1 = max(a, b)
		if min_b_from_i[a1] == b1:
			count_b[a1] += 1

	min1 = [INF] * (n + 2)
	min2 = [INF] * (n + 2)
	count_min = [0] * (n + 2)

	min1[n] = min_b_from_i[n]
	if min_b_from_i[n] != INF:
		count_min[n] = count_b[n]
	else:
		count_min[n] = 0
	min2[n] = INF

	for i in range(n - 1, 0, -1):
		candidates = set()
		if min_b_from_i[i] != INF:
			candidates.add(min_b_from_i[i])
		if min1[i + 1] != INF:
			candidates.add(min1[i + 1])
		if min2[i + 1] != INF:
			candidates.add(min2[i + 1])
		if not candidates:
			min1[i] = INF
			min2[i] = INF
			count_min[i] = 0
		else:
			sorted_candidates = sorted(candidates)
			min1[i] = sorted_candidates[0]
			if len(sorted_candidates) >= 2:
				min2[i] = sorted_candidates[1]
			else:
				min2[i] = INF
			count = 0
			if min_b_from_i[i] == min1[i]:
				count += count_b[i]
			if min1[i + 1] == min1[i]:
				count += count_min[i + 1]
			count_min[i] = count

	val = [0] * (n + 2)
	for i in range(1, n + 1):
		if count_min[i] == 1 and min1[i] != INF:
			if min2[i] == INF:
				val[i] = n - min1[i] + 1
			else:
				val[i] = min2[i] - min1[i]
		else:
			val[i] = 0

	by_b = defaultdict(list)
	for i in range(1, n + 1):
		if count_min[i] == 1 and min1[i] != INF:
			b_val = min1[i]
			by_b[b_val].append((i, val[i]))
	
	prefix_by_b = {}
	for b_val, lst in by_b.items():
		lst.sort(key=lambda x: x[0])
		prefix = [0]
		for idx, (i_val, v_val) in enumerate(lst):
			prefix.append(prefix[-1] + v_val)
		prefix_by_b[b_val] = (lst, prefix)
	
	base = total_subarrays - total_union
	best = 0
	for edge in conflictingPairs:
		a, b = edge
		a0 = min(a, b)
		b0 = max(a, b)
		if b0 in prefix_by_b:
			lst, prefix_list = prefix_by_b[b0]
			lo, hi = 0, len(lst) - 1
			pos = -1
			while lo <= hi:
				mid = (lo + hi) // 2
				if lst[mid][0] <= a0:
					pos = mid
					lo = mid + 1
				else:
					hi = mid - 1
			if pos == -1:
				Xe = 0
			else:
				Xe = prefix_list[pos + 1]
		else:
			Xe = 0
		candidate = base + Xe
		if candidate > best:
			best = candidate
	print(best)

if __name__ == "__main__":
	main()