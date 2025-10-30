import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	points = []
	index = 1
	for i in range(n):
		x = int(data[index])
		y = int(data[index + 1])
		index += 2
		points.append((x, y))
	
	group0 = []
	group1 = []
	
	for (x, y) in points:
		if (x + y) % 2 == 0:
			group0.append((x, y))
		else:
			group1.append((x, y))
			
	total = 0
	
	def sum_abs_diff(arr):
		arr.sort()
		total_sum = 0
		prefix = 0
		for i, val in enumerate(arr):
			total_sum += val * i - prefix
			prefix += val
		return total_sum
	
	if len(group0) >= 2:
		u_list = [x + y for (x, y) in group0]
		v_list = [x - y for (x, y) in group0]
		total += (sum_abs_diff(u_list) + sum_abs_diff(v_list)) // 2
		
	if len(group1) >= 2:
		u_list = [x + y for (x, y) in group1]
		v_list = [x - y for (x, y) in group1]
		total += (sum_abs_diff(u_list) + sum_abs_diff(v_list)) // 2
		
	print(total)

if __name__ == '__main__':
	main()