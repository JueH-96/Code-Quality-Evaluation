def main():
	A, M, L, R = map(int, input().split())
	count = (R - A) // M - (L - 1 - A) // M
	print(count)

if __name__ == '__main__':
	main()