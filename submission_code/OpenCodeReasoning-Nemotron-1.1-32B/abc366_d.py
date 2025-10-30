import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	
	A = [[[0] * (N+1) for _ in range(N+1)] for __ in range(N+1)]
	
	for x in range(1, N+1):
		for y in range(1, N+1):
			row = [int(next(it)) for _ in range(N)]
			for z in range(1, N+1):
				A[x][y][z] = row[z-1]
				
	P = [[[0] * (N+1) for _ in range(N+1)] for __ in range(N+1)]
	
	for i in range(1, N+1):
		for j in range(1, N+1):
			for k in range(1, N+1):
				P[i][j][k] = A[i][j][k] + P[i-1][j][k] + P[i][j-1][k] + P[i][j][k-1] - P[i-1][j-1][k] - P[i-1][j][k-1] - P[i][j-1][k-1] + P[i-1][j-1][k-1]
				
	Q = int(next(it))
	out_lines = []
	for _ in range(Q):
		Lx = int(next(it)); Rx = int(next(it))
		Ly = int(next(it)); Ry = int(next(it))
		Lz = int(next(it)); Rz = int(next(it))
		
		total = P[Rx][Ry][Rz] 
		total -= P[Lx-1][Ry][Rz] 
		total -= P[Rx][Ly-1][Rz] 
		total -= P[Rx][Ry][Lz-1] 
		total += P[Lx-1][Ly-1][Rz] 
		total += P[Lx-1][Ry][Lz-1] 
		total += P[Rx][Ly-1][Lz-1] 
		total -= P[Lx-1][Ly-1][Lz-1]
		
		out_lines.append(str(total))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()