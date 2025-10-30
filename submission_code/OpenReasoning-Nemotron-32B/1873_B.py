t = int(input().strip())
for _ in range(t):
	n_line = input().strip()
	while n_line == '':
		n_line = input().strip()
	n = int(n_line)
	
	arr_line = input().strip()
	while arr_line == '':
		arr_line = input().strip()
	arr = list(map(int, arr_line.split()))
	
	best = 0
	for i in range(n):
		new_arr = arr.copy()
		new_arr[i] += 1
		prod = 1
		for num in new_arr:
			prod *= num
		if prod > best:
			best = prod
	print(best)