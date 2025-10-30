n = int(input().strip())
candidate1 = (n // 5) * 5
candidate2 = candidate1 + 5
if n % 5 < 3:
	print(candidate1)
else:
	print(candidate2)