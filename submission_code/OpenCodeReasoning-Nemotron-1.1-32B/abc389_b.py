X = int(input().strip())

fact = 1
for n in range(1, 21):
	fact *= n
	if fact == X:
		print(n)
		break