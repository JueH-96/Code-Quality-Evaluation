def main():
	A, B = map(int, input().split())
	n = (A + B - 1) // B
	print(n)

if __name__ == '__main__':
	main()