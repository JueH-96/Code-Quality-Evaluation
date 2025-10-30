import sys

def main():
	n = int(sys.stdin.readline().strip())
	min_dict = {}
	for _ in range(n):
		a, c = map(int, sys.stdin.readline().split())
		if c in min_dict:
			if a < min_dict[c]:
				min_dict[c] = a
		else:
			min_dict[c] = a
	print(max(min_dict.values()))

if __name__ == '__main__':
	main()