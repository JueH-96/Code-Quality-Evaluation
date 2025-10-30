def main():
	n, k = map(int, input().split())
	arr = list(map(int, input().split()))
	res = [str(num // k) for num in arr if num % k == 0]
	print(" ".join(res))

if __name__ == "__main__":
	main()