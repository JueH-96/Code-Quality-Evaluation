import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	i = 0
	total = 0
	for j in range(n):
		while i < n and A[i] <= A[j] // 2:
			i += 1
		total += i
	print(total)

if __name__ == '__main__':
	main()