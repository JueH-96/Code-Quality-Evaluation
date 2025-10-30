def main():
	n = int(input().strip())
	results = []
	for i in range(1, n + 1):
		if i % 3 == 0:
			results.append('x')
		else:
			results.append('o')
	print(''.join(results))

if __name__ == "__main__":
	main()