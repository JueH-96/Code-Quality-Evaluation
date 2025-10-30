import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	A.sort()
	B.sort()
	
	j = 0
	total_cost = 0
	for i in range(m):
		while j < n and A[j] < B[i]:
			j += 1
		if j >= n:
			print(-1)
			return
		total_cost += A[j]
		j += 1
		
	print(total_cost)

if __name__ == "__main__":
	main()