def main():
	n = int(input().strip())
	distinct = set()
	for _ in range(n):
		s = input().strip()
		rev = s[::-1]
		canonical = min(s, rev)
		distinct.add(canonical)
	print(len(distinct))

if __name__ == '__main__':
	main()