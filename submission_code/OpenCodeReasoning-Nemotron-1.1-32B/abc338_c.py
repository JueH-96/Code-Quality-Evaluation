import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	Q = list(map(int, data[1:1+n]))
	A = list(map(int, data[1+n:1+2*n]))
	B = list(map(int, data[1+2*n:1+3*n]))
	
	x_max = 10**18
	for i in range(n):
		if A[i] > 0:
			x_candidate = Q[i] // A[i]
			if x_candidate < x_max:
				x_max = x_candidate
				
	best = 0
	for x in range(0, x_max + 1):
		y_ub = 10**18
		for i in range(n):
			if A[i] == 0 and B[i] == 0:
				continue
			if A[i] == 0 and B[i] > 0:
				candidate = Q[i] // B[i]
				if candidate < y_ub:
					y_ub = candidate
			elif A[i] > 0 and B[i] > 0:
				rem = Q[i] - x * A[i]
				if rem < 0:
					candidate = -10**18
				else:
					candidate = rem // B[i]
				if candidate < y_ub:
					y_ub = candidate
			if y_ub < 0:
				break
		total = x + max(0, y_ub)
		if total > best:
			best = total
			
	print(best)

if __name__ == "__main__":
	main()