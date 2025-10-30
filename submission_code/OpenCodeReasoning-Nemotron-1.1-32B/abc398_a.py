n = int(input().strip())
if n % 2 == 0:
	part = '-' * (n // 2 - 1)
	s = part + '==' + part
else:
	part = '-' * ((n - 1) // 2)
	s = part + '=' + part
print(s)