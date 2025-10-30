def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	result = []
	for i in range(n - 1):
		result.append(str(arr[i] * arr[i + 1]))
	print(" ".join(result))

if __name__ == "__main__":
	main()