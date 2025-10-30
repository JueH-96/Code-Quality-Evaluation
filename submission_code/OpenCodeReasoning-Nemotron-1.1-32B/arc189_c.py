import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	N = int(next(it))
	X = int(next(it))
	X0 = X - 1
	
	A = [int(next(it)) for _ in range(N)]
	B = [int(next(it)) for _ in range(N)]
	P = [int(next(it)) - 1 for _ in range(N)]
	Q = [int(next(it)) - 1 for _ in range(N)]
	
	cycle_red = set()
	cur = X0
	while True:
		cycle_red.add(cur)
		cur = P[cur]
		if cur == X0:
			break
	cycle_red.discard(X0)
	
	cycle_blue = set()
	cur = X0
	while True:
		cycle_blue.add(cur)
		cur = Q[cur]
		if cur == X0:
			break
	cycle_blue.discard(X0)
	
	for i in range(N):
		if i != X0 and A[i] == 1:
			if i not in cycle_red:
				print(-1)
				return
				
	for i in range(N):
		if i != X0 and B[i] == 1:
			if i not in cycle_blue:
				print(-1)
				return
				
	has_red = any(A[i] for i in range(N) if i != X0)
	has_blue = any(B[i] for i in range(N) if i != X0)
	
	R = cycle_red if has_red else set()
	B_set = cycle_blue if has_blue else set()
	
	total_ops = len(R | B_set)
	print(total_ops)

if __name__ == '__main__':
	main()