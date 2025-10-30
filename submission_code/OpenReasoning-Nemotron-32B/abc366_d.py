import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	index = 1
	A = [[[0] * (n+1) for _ in range(n+1)] for __ in range(n+1)]
	
	for x in range(1, n+1):
		for y in range(1, n+1):
			for z in range(1, n+1):
				A[x][y][z] = int(data[index])
				index += 1
	
	P = [[[0] * (n+1) for _ in range(n+1)] for __ in range(n+1)]
	
	for i in range(1, n+1):
		for j in range(1, n+1):
			for k in range(1, n+1):
				P[i][j][k] = A[i][j][k] + P[i-1][j][k] + P[i][j-1][k] + P[i][j][k-1] \
							  - P[i-1][j-1][k] - P[i-1][j][k-1] - P[i][j-1][k-1] \
							  + P[i-1][j-1][k-1]
	
	q = int(data[index])
	index += 1
	out_lines = []
	for _ in range(q):
		Lx = int(data[index])
		Rx = int(data[index+1])
		Ly = int(data[index+2])
		Ry = int(data[index+3])
		Lz = int(data[index+4])
		Rz = int(data[index+5])
		index += 6
		total = P[Rx][Ry][Rz] \
				- P[Lx-1][Ry][Rz] - P[Rx][Ly-1][Rz] - P[Rx][Ry][Lz-1] \
				+ P[Lx-1][Ly-1][Rz] + P[Lx-1][Ry][Lz-1] + P[Rx][Ly-1][Lz-1] \
				- P[Lx-1][Ly-1][Lz-1]
		out_lines.append(str(total))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()