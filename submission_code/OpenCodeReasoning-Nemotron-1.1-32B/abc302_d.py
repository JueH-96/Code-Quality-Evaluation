import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	D = int(data[2])
	A = list(map(int, data[3:3+n]))
	B = list(map(int, data[3+n:3+n+m]))
	
	A.sort()
	B.sort()
	
	j = m - 1
	best = -1
	
	for i in range(n-1, -1, -1):
		while j >= 0 and B[j] > A[i] + D:
			j -= 1
		if j >= 0 and B[j] >= A[i] - D:
			total = A[i] + B[j]
			if total > best:
				best = total
				
	print(best)

if __name__ == '__main__':
	main()