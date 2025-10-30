import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	D = int(data[1])
	xs = []
	ys = []
	index = 2
	for i in range(n):
		x = int(data[index])
		y = int(data[index+1])
		index += 2
		xs.append(x)
		ys.append(y)
	
	xs_sorted = sorted(xs)
	total_x = sum(xs)
	prefix_x = [0]
	for num in xs_sorted:
		prefix_x.append(prefix_x[-1] + num)
	
	ys_sorted = sorted(ys)
	total_y = sum(ys)
	prefix_y = [0]
	for num in ys_sorted:
		prefix_y.append(prefix_y[-1] + num)
	
	def gx(x):
		left_count = bisect.bisect_right(xs_sorted, x)
		left_sum = prefix_x[left_count]
		right_count = n - left_count
		right_sum = total_x - left_sum
		return (left_count * x - left_sum) + (right_sum - right_count * x)
	
	def hy(y):
		left_count = bisect.bisect_right(ys_sorted, y)
		left_sum = prefix_y[left_count]
		right_count = n - left_count
		right_sum = total_y - left_sum
		return (left_count * y - left_sum) + (right_sum - right_count * y)
	
	min_x = xs_sorted[0]
	max_x = xs_sorted[-1]
	med_x = xs_sorted[n//2]
	
	if gx(med_x) > D:
		Lx = Rx = None
	else:
		low = min_x - D - 1
		high = med_x
		while high - low > 1:
			mid = (low + high) // 2
			if gx(mid) <= D:
				high = mid
			else:
				low = mid
		Lx = high
		
		low = med_x
		high = max_x + D + 1
		while high - low > 1:
			mid = (low + high) // 2
			if gx(mid) <= D:
				low = mid
			else:
				high = mid
		Rx = low
	
	min_y = ys_sorted[0]
	max_y = ys_sorted[-1]
	med_y = ys_sorted[n//2]
	
	if hy(med_y) > D:
		Ly = Ry = None
	else:
		low = min_y - D - 1
		high = med_y
		while high - low > 1:
			mid = (low + high) // 2
			if hy(mid) <= D:
				high = mid
			else:
				low = mid
		Ly = high
		
		low = med_y
		high = max_y + D + 1
		while high - low > 1:
			mid = (low + high) // 2
			if hy(mid) <= D:
				low = mid
			else:
				high = mid
		Ry = low
	
	if Lx is None or Ly is None:
		print(0)
		return
	
	arr_h = []
	for y in range(Ly, Ry + 1):
		arr_h.append(hy(y))
	
	arr_h.sort()
	
	total_pairs = 0
	for x in range(Lx, Rx + 1):
		T = D - gx(x)
		if T < 0:
			continue
		count = bisect.bisect_right(arr_h, T)
		total_pairs += count
	
	print(total_pairs)

if __name__ == "__main__":
	main()