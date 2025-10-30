import sys

def main():
	n = int(sys.stdin.readline().strip())
	A = list(map(int, sys.stdin.readline().split()))
	B = list(map(int, sys.stdin.readline().split()))
	maxA = max(A)
	maxB = max(B)
	print(maxA + maxB)

if __name__ == '__main__':
	main()