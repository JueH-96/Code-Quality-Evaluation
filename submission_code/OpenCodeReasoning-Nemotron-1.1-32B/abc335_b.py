n = int(input().strip())
for x in range(n + 1):
	for y in range(n - x + 1):
		for z in range(n - x - y + 1):
			print(f"{x} {y} {z}")