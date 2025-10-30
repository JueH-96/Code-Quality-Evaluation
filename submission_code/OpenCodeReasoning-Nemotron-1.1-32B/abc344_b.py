numbers = []
while True:
	num = int(input().strip())
	numbers.append(num)
	if num == 0:
		break

numbers.reverse()
for num in numbers:
	print(num)