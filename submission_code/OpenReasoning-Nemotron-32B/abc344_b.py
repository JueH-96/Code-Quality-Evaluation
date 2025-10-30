import sys

numbers = []
for line in sys.stdin:
	stripped = line.strip()
	if stripped == '':
		continue
	num = int(stripped)
	numbers.append(num)
	if num == 0:
		break

for num in reversed(numbers):
	print(num)