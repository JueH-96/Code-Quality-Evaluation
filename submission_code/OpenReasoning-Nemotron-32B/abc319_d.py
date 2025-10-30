import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	L = list(map(int, data[2:2+n]))
	
	total = sum(L) + (n - 1)
	max_L = max(L)
	
	def feasible(W):
		lines = 1
		current = L[0]
		for i in range(1, n):
			if current + 1 + L[i] <= W:
				current += 1 + L[i]
			else:
				lines += 1
				if lines > m:
					return False
				current = L[i]
		return True

	low, high = max_L, total
	while low < high:
		mid = (low + high) // 2
		if feasible(mid):
			high = mid
		else:
			low = mid + 1

	print(low)

if __name__ == '__main__':
	main()