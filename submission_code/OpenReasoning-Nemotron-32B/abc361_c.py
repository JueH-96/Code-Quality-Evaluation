import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	L = n - k

	A.sort()
	low = 0
	high = A[-1] - A[0]

	def check(mid):
		j = 0
		n_val = len(A)
		for i in range(n_val):
			while j < n_val and A[j] - A[i] <= mid:
				j += 1
			if j - i >= L:
				return True
		return False

	while low < high:
		mid = (low + high) // 2
		if check(mid):
			high = mid
		else:
			low = mid + 1

	print(low)

if __name__ == '__main__':
	main()