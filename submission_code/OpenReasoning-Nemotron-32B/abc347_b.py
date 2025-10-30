def main():
	s = input().strip()
	n = len(s)
	unique_substrings = set()
	for i in range(n):
		for j in range(i + 1, n + 1):
			unique_substrings.add(s[i:j])
	print(len(unique_substrings))

if __name__ == "__main__":
	main()