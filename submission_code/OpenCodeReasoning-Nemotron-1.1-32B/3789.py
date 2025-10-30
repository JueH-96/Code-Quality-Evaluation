import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	conflictingPairs = []
	index = 1
	for i in range(len(data) // 2 - 1):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		conflictingPairs.append([a, b])
	
	total_subarrays = n * (n + 1) // 2
	
	min_r = [n] * n
	add = [[] for _ in range(n)]
	for a, b in conflictingPairs:
		i1, i2 = a - 1, b - 1
		L0 = min(i1, i2)
		R0 = max(i1, i2)
		if L0 < n:
			add[L0].append(R0)
	
	cur_min = n
	for l in range(n - 1, -1, -1):
		if add[l]:
			cur_min = min(cur_min, min(add[l]))
		min_r[l] = cur_min
	
	total_valid = 0
	for l in range(n):
		total_valid += (min_r[l] - l)
	
	ans = total_valid
	for a, b in conflictingPairs:
		L0 = min(a - 1, b - 1)
		R0 = max(a - 1, b - 1)
		count_p = (L0 + 1) * (n - R0)
		
		type1 = []
		type2 = []
		type3 = []
		type4_exists = False
		for a2, b2 in conflictingPairs:
			if a2 == a and b2 == b:
				continue
			i1, i2 = a2 - 1, b2 - 1
			L0q = min(i1, i2)
			R0q = max(i1, i2)
			if L0q <= L0 and R0q >= R0:
				type1.append((L0q, R0q))
			elif L0q > L0 and R0q >= R0:
				type2.append(R0q)
			elif L0q <= L0 and R0q < R0:
				type3.append(L0q)
			elif L0q > L0 and R0q < R0:
				type4_exists = True
		
		if type4_exists:
			candidate = total_valid
		else:
			low_bound = 0
			if type3:
				low_bound = max(type3)
			high_bound = n - 1
			if type2:
				high_bound = min(type2) - 1
			if low_bound >= L0 + 1 or high_bound < R0:
				candidate = total_valid
			else:
				l_min = low_bound + 1
				r_max = high_bound
				if l_min > L0 or r_max < R0:
					candidate = total_valid
				else:
					region_l_min = l_min
					region_l_max = L0
					region_r_min = R0
					region_r_max = r_max
					count_region = (region_l_max - region_l_min + 1) * (region_r_max - region_r_min + 1)
					for L0q, R0q in type1:
						if region_l_min <= L0q and region_r_max >= R0q:
							continue
						if region_l_max <= L0q or region_r_min >= R0q:
							continue
						low1 = max(region_l_min, L0q + 1)
						high1 = region_r_max
						count1 = 0
						if low1 <= region_l_max:
							count1 = (region_l_max - low1 + 1) * (region_r_max - region_r_min + 1)
						low2 = region_l_min
						high2 = min(region_r_max, R0q - 1)
						count2 = 0
						if high2 >= region_r_min:
							count2 = (region_l_max - region_l_min + 1) * (high2 - region_r_min + 1)
						low3 = max(region_l_min, L0q + 1)
						high3 = min(region_r_max, R0q - 1)
						count3 = 0
						if low3 <= region_l_max and high3 >= region_r_min:
							count3 = (region_l_max - low3 + 1) * (high3 - region_r_min + 1)
						count_region -= (count1 + count2 - count3)
					candidate = total_valid + count_region
		if candidate > ans:
			ans = candidate
	print(ans)

if __name__ == '__main__':
	main()