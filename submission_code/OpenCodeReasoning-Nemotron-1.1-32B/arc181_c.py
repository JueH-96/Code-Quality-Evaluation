import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	P = list(map(int, data[1:1+n]))
	Q = list(map(int, data[1+n:1+2*n]))
	
	row_order = [0] * n
	for idx in range(n):
		row_order[P[idx]-1] = idx
		
	col_order = [0] * n
	for idx in range(n):
		col_order[Q[idx]-1] = idx
		
	for i in range(n-1, -1, -1):
		s = ''
		for j in range(n):
			if row_order[i] < col_order[j]:
				s += '1'
			else:
				s += '0'
		print(s)

if __name__ == '__main__':
	main()