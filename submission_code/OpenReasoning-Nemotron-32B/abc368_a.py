def main():
	n, k = map(int, input().split())
	a = list(map(int, input().split()))
	result = a[n - k:] + a[:n - k]
	print(" ".join(map(str, result)))

if __name__ == '__main__':
	main()