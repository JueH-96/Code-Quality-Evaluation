def main():
	n, k = map(int, input().split())
	a = list(map(int, input().split()))
	quotients = [str(num // k) for num in a if num % k == 0]
	print(" ".join(quotients))

if __name__ == '__main__':
	main()