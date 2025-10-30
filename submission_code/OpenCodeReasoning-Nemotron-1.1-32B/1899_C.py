t = int(input().strip())
for _ in range(t):
	n = int(input().strip())
	a = list(map(int, input().split()))
	current = a[0]
	max_sum = a[0]
	for i in range(1, n):
		if (a[i] % 2) != (a[i-1] % 2):
			current = max(a[i], current + a[i])
		else:
			current = a[i]
		if current > max_sum:
			max_sum = current
	print(max_sum)