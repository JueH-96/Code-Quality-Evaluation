n = int(input().strip())
arr = list(map(int, input().split()))

if n == 2:
	print("Yes")
else:
	for i in range(1, n-1):
		if arr[i] * arr[i] != arr[i-1] * arr[i+1]:
			print("No")
			break
	else:
		print("Yes")