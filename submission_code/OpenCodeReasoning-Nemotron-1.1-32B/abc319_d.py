import sys

def feasible(W, L, M):
	lines = 1
	current = 0
	for word in L:
		if current == 0:
			current = word
		else:
			if current + 1 + word <= W:
				current += 1 + word
			else:
				lines += 1
				current = word
		if lines > M:
			return False
	return True

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	M = int(data[1])
	L = list(map(int, data[2:2+n]))
	
	low = max(L)
	high = sum(L) + (n - 1)
	
	while low < high:
		mid = (low + high) // 2
		if feasible(mid, L, M):
			high = mid
		else:
			low = mid + 1
			
	print(low)

if __name__ == '__main__':
	main()