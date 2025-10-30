n = int(input().strip())
heights = list(map(int, input().split()))

for i in range(1, n):
	if heights[i] > heights[0]:
		print(i + 1)
		break
else:
	print(-1)