import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	N = int(next(it))
	M = int(next(it))
	X = [int(next(it)) for _ in range(M)]
	A = [int(next(it)) for _ in range(M)]
	
	total_stones = sum(A)
	if total_stones != N:
		print(-1)
		return
		
	stones = list(zip(X, A))
	stones.sort()
	
	prev = 0
	current = 0
	for x, a in stones:
		gap = x - prev - 1
		if current < gap:
			print(-1)
			return
		current -= gap
		current += a - 1
		prev = x
		
	total_final = N * (N + 1) // 2
	total_initial = 0
	for x, a in stones:
		total_initial += x * a
		
	ans = total_final - total_initial
	print(ans)

if __name__ == '__main__':
	main()