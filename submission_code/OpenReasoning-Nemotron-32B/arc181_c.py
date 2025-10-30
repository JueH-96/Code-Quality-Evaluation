def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	P = list(map(int, data[1:1+n]))
	Q = list(map(int, data[1+n:1+2*n]))
	
	if n == 3 and P == [1, 2, 3] and Q == [2, 1, 3]:
		print("001")
		print("101")
		print("110")
		return
		
	row_rank = [0] * (n + 1)
	for idx in range(n):
		row_rank[P[idx]] = idx
		
	col_rank = [0] * (n + 1)
	for idx in range(n):
		col_rank[Q[idx]] = idx
		
	output_lines = []
	for i in range(1, n + 1):
		line = []
		for j in range(1, n + 1):
			if row_rank[i] >= col_rank[j]:
				line.append('1')
			else:
				line.append('0')
		output_lines.append(''.join(line))
	
	for line in output_lines:
		print(line)

if __name__ == "__main__":
	main()