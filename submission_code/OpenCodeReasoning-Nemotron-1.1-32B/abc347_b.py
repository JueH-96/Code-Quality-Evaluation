def main():
	s = input().strip()
	n = len(s)
	distinct_substrings = set()
	for i in range(n):
		for j in range(i, n):
			substr = s[i:j+1]
			distinct_substrings.add(substr)
	print(len(distinct_substrings))

if __name__ == "__main__":
	main()