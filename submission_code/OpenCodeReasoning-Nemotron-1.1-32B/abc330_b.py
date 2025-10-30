import sys

def main():
	data = sys.stdin.readline().split()
	n = int(data[0])
	L = int(data[1])
	R = int(data[2])
	A = list(map(int, sys.stdin.readline().split()))
	res = " ".join(str(L) if a < L else str(R) if a > R else str(a) for a in A)
	print(res)

if __name__ == "__main__":
	main()