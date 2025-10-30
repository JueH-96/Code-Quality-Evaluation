def main():
	n, k, x = map(int, input().split())
	A = list(map(int, input().split()))
	B = A[:k] + [x] + A[k:]
	print(" ".join(map(str, B)))

if __name__ == "__main__":
	main()