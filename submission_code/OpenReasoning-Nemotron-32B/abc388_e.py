def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	i = 0
	for j in range(n // 2, n):
		if 2 * arr[i] <= arr[j]:
			i += 1
	print(i)

if __name__ == '__main__':
	main()