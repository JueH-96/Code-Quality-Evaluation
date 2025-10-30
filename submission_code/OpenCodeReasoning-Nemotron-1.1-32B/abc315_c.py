n = int(input().strip())
top1 = {}
top2 = {}
for _ in range(n):
	f, s = map(int, input().split())
	if f not in top1:
		top1[f] = s
	else:
		if s > top1[f]:
			top2[f] = top1[f]
			top1[f] = s
		else:
			if f in top2:
				if s > top2[f]:
					top2[f] = s
			else:
				top2[f] = s

candidate_same = 0
for f in top1:
	if f in top2:
		cand = top1[f] + top2[f] // 2
		if cand > candidate_same:
			candidate_same = cand

max_vals = list(top1.values())
if len(max_vals) < 2:
	print(candidate_same)
else:
	best1 = -1
	best2 = -1
	for val in max_vals:
		if val > best1:
			best2 = best1
			best1 = val
		elif val > best2:
			best2 = val
	candidate_diff = best1 + best2
	ans = max(candidate_same, candidate_diff)
	print(ans)