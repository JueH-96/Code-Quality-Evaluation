def main():
	n = int(input().strip())
	s = input().strip()
	seen = set()
	for i, char in enumerate(s):
		seen.add(char)
		if len(seen) == 3:
			print(i + 1)
			return
	print(n)

if __name__ == '__main__':
	main()