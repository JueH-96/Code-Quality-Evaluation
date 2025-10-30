import bisect

def main():
	n = int(input().strip())
	K = list(map(int, input().split()))
	total = sum(K)
	
	mid = n // 2
	left_arr = K[:mid]
	right_arr = K[mid:]
	
	left_sums = {0}
	for num in left_arr:
		new_sums = set()
		for s in left_sums:
			new_sums.add(s + num)
		left_sums.update(new_sums)
	
	right_sums = {0}
	for num in right_arr:
		new_sums = set()
		for s in right_sums:
			new_sums.add(s + num)
		right_sums.update(new_sums)
	
	left_sums = sorted(left_sums)
	right_sums = sorted(right_sums)
	
	best = total
	
	for s1 in left_sums:
		target = total / 2.0 - s1
		idx = bisect.bisect_left(right_sums, target)
		
		if idx < len(right_sums):
			s2 = right_sums[idx]
			total_sum = s1 + s2
			candidate = max(total_sum, total - total_sum)
			if candidate < best:
				best = candidate
				
		if idx > 0:
			s2 = right_sums[idx-1]
			total_sum = s1 + s2
			candidate = max(total_sum, total - total_sum)
			if candidate < best:
				best = candidate
				
	print(best)

if __name__ == '__main__':
	main()