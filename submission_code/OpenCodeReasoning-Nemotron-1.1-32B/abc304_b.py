n = int(input().strip())
if n < 1000:
	print(n)
else:
	if n < 10000:
		divisor = 10
	elif n < 100000:
		divisor = 100
	elif n < 1000000:
		divisor = 1000
	elif n < 10000000:
		divisor = 10000
	elif n < 100000000:
		divisor = 100000
	else:
		divisor = 1000000
	result = (n // divisor) * divisor
	print(result)