n, m, p = map(int, input().split())
if m > n:
	print(0)
else:
	print((n - m) // p + 1)