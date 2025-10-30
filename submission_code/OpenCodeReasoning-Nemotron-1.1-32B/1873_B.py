t = int(input().strip())
for _ in range(t):
	n_line = input().strip()
	while n_line == '':
		n_line = input().strip()
	n = int(n_line)
	a_line = input().split()
	while len(a_line) == 0:
		a_line = input().split()
	a = list(map(int, a_line))
	
	max_prod = 0
	for i in range(n):
		b = a.copy()
		b[i] = a[i] + 1
		prod = 1
		for num in b:
			prod *= num
		if prod > max_prod:
			max_prod = prod
			
	print(max_prod)