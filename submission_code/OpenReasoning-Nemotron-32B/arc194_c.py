import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+2*n]))
	C = list(map(int, data[1+2*n:1+3*n]))
	
	T0 = 0
	for i in range(n):
		T0 += A[i] * C[i]
	
	d_list = []
	for i in range(n):
		if A[i] != B[i]:
			d_list.append(C[i] - 2 * A[i] * C[i])
			
	m = len(d_list)
	d_list.sort()
	
	total_cost = T0 * m
	for i in range(m):
		total_cost += d_list[i] * (m - i)
		
	print(total_cost)

if __name__ == '__main__':
	main()