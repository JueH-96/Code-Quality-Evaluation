import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	X = list(map(int, data[2:2+m]))
	A = list(map(int, data[2+m:2+2*m]))
	
	total_stones = sum(A)
	if total_stones != n:
		print(-1)
		return
		
	stones = list(zip(X, A))
	stones.sort()
	
	if stones[0][0] != 1:
		print(-1)
		return
		
	cum = 0
	for i in range(m):
		x, a = stones[i]
		cum += a
		if i < m-1:
			next_x = stones[i+1][0]
			if cum < next_x - 1:
				print(-1)
				return
				
	cum = 0
	total_ops = 0
	for i in range(m):
		x, a = stones[i]
		T = cum + 1
		total_ops += a * (T - x) + a * (a - 1) // 2
		cum += a
		
	print(total_ops)

if __name__ == "__main__":
	main()