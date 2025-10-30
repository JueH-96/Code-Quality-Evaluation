import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index]); index += 1
		A = list(map(int, data[index:index+2*N]))
		index += 2*N
		total = 2 * N
		
		pos = [[] for _ in range(N+1)]
		for idx, num in enumerate(A):
			if 1 <= num <= N:
				pos[num].append(idx)
		
		is_valid = [False] * (N+1)
		for i in range(1, N+1):
			if len(pos[i]) != 2:
				continue
			p0, p1 = pos[i]
			if abs(p0 - p1) != 1:
				is_valid[i] = True
				
		count = 0
		for a in range(1, N+1):
			if not is_valid[a]:
				continue
			positions_a = pos[a]
			if len(positions_a) != 2:
				continue
			p, q = sorted(positions_a)
			
			if q + 1 < total:
				if A[p+1] == A[q+1] and A[p+1] != a:
					b = A[p+1]
					if 1 <= b <= N:
						positions_b = pos[b]
						if len(positions_b) == 2:
							b_sorted = sorted(positions_b)
							if b_sorted == [p+1, q+1] and is_valid[b]:
								if a < b:
									count += 1
									
			if q - p != 2:
				if p+1 < total and q-1 >= 0:
					if A[p+1] == A[q-1] and A[p+1] != a:
						b = A[p+1]
						if 1 <= b <= N:
							positions_b = pos[b]
							if len(positions_b) == 2:
								b_sorted = sorted(positions_b)
								if b_sorted == [p+1, q-1] and is_valid[b]:
									if a < b:
										count += 1
										
		results.append(str(count))
	
	print("
".join(results))

if __name__ == "__main__":
	main()