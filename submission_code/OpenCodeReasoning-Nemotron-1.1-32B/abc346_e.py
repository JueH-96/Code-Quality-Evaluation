import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	H = int(data[0])
	W = int(data[1])
	M = int(data[2])
	index = 3
	
	row_time = [-1] * (H + 1)
	row_color = [0] * (H + 1)
	col_time = [-1] * (W + 1)
	col_color = [0] * (W + 1)
	
	for idx in range(M):
		t = int(data[index])
		a = int(data[index + 1])
		x = int(data[index + 2])
		index += 3
		if t == 1:
			row_time[a] = idx
			row_color[a] = x
		else:
			col_time[a] = idx
			col_color[a] = x
			
	row_times_list = row_time[1:]
	row_times_list.sort()
	
	col_times_list = col_time[1:]
	col_times_list.sort()
	
	max_color_val = 200000
	freq = [0] * (max_color_val + 1)
	
	for i in range(1, H + 1):
		t_val = row_time[i]
		if t_val == -1:
			count = 0
		else:
			pos = bisect.bisect_left(col_times_list, t_val)
			count = pos
		color_val = row_color[i]
		if color_val <= max_color_val:
			freq[color_val] += count
			
	for j in range(1, W + 1):
		t_val = col_time[j]
		pos = bisect.bisect_right(row_times_list, t_val)
		count = pos
		color_val = col_color[j]
		if color_val <= max_color_val:
			freq[color_val] += count
			
	distinct_colors = []
	for color in range(max_color_val + 1):
		if freq[color] > 0:
			distinct_colors.append(color)
			
	print(len(distinct_colors))
	for color in distinct_colors:
		print(f"{color} {freq[color]}")

if __name__ == "__main__":
	main()