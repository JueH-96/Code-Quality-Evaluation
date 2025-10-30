import sys

def feasible(S, Q, A, B):
	n = len(Q)
	x_lower = 0
	x_upper = S
	for i in range(n):
		d = A[i] - B[i]
		num = Q[i] - S * B[i]
		if d == 0:
			if num < 0:
				return False
		elif d > 0:
			if num < 0:
				return False
			x_upper = min(x_upper, num // d)
		else:
			if num < 0:
				num_abs = -num
				d_abs = -d
				bound = (num_abs + d_abs - 1) // d_abs
				x_lower = max(x_lower, bound)
		if x_lower > x_upper:
			return False
	return x_lower <= x_upper

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	Q = list(map(int, data[1:1+n]))
	A = list(map(int, data[1+n:1+2*n]))
	B = list(map(int, data[1+2*n:1+3*n]))
	
	low = 0
	high = 10**7
	
	while low <= high:
		mid = (low + high) // 2
		if feasible(mid, Q, A, B):
			low = mid + 1
		else:
			high = mid - 1
			
	print(high)

if __name__ == '__main__':
	main()