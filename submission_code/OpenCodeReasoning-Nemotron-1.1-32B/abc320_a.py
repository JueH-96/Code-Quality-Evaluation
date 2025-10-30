def main():
	A, B = map(int, input().split())
	result = A ** B + B ** A
	print(result)

if __name__ == "__main__":
	main()