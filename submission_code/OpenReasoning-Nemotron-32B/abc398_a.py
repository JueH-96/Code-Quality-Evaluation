n = int(input().strip())
if n % 2 == 1:
	k = (n - 1) // 2
	print('-' * k + '=' + '-' * k)
else:
	k = (n - 2) // 2
	print('-' * k + '==' + '-' * k)