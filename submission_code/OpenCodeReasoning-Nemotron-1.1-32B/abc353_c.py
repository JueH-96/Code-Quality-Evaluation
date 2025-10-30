mod = 10**8
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	total_sum = sum(A)
	B = sorted(A)
	j = n - 1
	count = 0
	for i in range(n):
		while j > i and B[i] + B[j] >= mod:
			j -= 1
		a = max(j + 1, i + 1)
		if a < n:
			count += (n - a)
	result = (n - 1) * total_sum - mod * count
	print(result)

if __name__ == '__main__':
	main()