import sys

def main():
	n = int(sys.stdin.readline().strip())
	distinct_set = set()
	for _ in range(n):
		s = sys.stdin.readline().strip()
		rev = s[::-1]
		if s <= rev:
			canonical = s
		else:
			canonical = rev
		distinct_set.add(canonical)
	print(len(distinct_set))

if __name__ == "__main__":
	main()