def main():
	n, l, r = map(int, input().split())
	arr = list(range(1, n + 1))
	arr[l-1:r] = arr[l-1:r][::-1]
	print(" ".join(map(str, arr)))

if __name__ == "__main__":
	main()