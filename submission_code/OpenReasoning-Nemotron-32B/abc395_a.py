n = int(input().strip())
arr = list(map(int, input().split()))

if all(arr[i] < arr[i+1] for i in range(n-1)):
	print('Yes')
else:
	print('No')