import bisect

def main():
	import sys
	data = sys.stdin.read().split()
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
	
	xs.sort()
	ys.sort()
	
	total_x = sum(xs)
	prefix_x = [0] * (n+1)
	for i in range(1, n+1):
		prefix_x[i] = prefix_x[i-1] + xs[i-1]
	
	total_y = sum(ys)
	prefix_y = [0] * (n+1)
	for i in range(1, n+1):
		prefix_y[i] = prefix_y[i-1] + ys[i-1]
	
	x_min = xs[0] - D
	x_max = xs[-1] + D
	y_min = ys[0] - D
	y_max = ys[-1] + D
	
	g_arr = []
	for y in range(y_min, y_max + 1):
		p = bisect.bisect_right(ys, y)
		sum_left = prefix_y[p]
		sum_right = total_y - sum_left
		count_left = p
		count_right = n - p
		g_val = y * count_left - sum_left + sum_right - y * count_right
		g_arr.append(g_val)
	
	g_arr.sort()
	
	ans = 0
	for x in range(x_min, x_max + 1):
		p = bisect.bisect_right(xs, x)
		sum_left = prefix_x[p]
		sum_right = total_x - sum_left
		count_left = p
		count_right = n - p
		f_val = x * count_left - sum_left + sum_right - x * count_right
		if f_val > D:
			continue
		rem = D - f_val
		count_y = bisect.bisect_right(g_arr, rem)
		ans += count_y
		
	print(ans)

if __name__ == "__main__":
	main()