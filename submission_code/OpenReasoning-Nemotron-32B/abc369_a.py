def main():
	A, B = map(int, input().split())
	candidates = set()
	candidates.add(2 * A - B)
	candidates.add(2 * B - A)
	if (A + B) % 2 == 0:
		candidates.add((A + B) // 2)
	print(len(candidates))

if __name__ == '__main__':
	main()