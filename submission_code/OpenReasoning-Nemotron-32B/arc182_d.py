import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	M = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+2*n]))
	
	total_ops = 0
	for i in range(n):
		a = A[i]
		b = B[i]
		if a == b:
			continue
			
		clockwise_dist = (b - a) % M
		if clockwise_dist < 0:
			clockwise_dist += M
		counter_dist = M - clockwise_dist
		d0 = min(clockwise_dist, counter_dist)
		
		f1 = B[i-1] if i > 0 else None
		f2 = A[i+1] if i < n-1 else None
		
		blocked_clockwise = True
		blocked_counterclockwise = True
		
		if clockwise_dist == d0:
			if a <= b:
				if (f1 is not None and a < f1 < b) or (f2 is not None and a < f2 < b):
					blocked_clockwise = True
				else:
					blocked_clockwise = False
			else:
				if (f1 is not None and (f1 > a or f1 < b)) or (f2 is not None and (f2 > a or f2 < b)):
					blocked_clockwise = True
				else:
					blocked_clockwise = False
					
		if counter_dist == d0:
			if a >= b:
				if (f1 is not None and b < f1 < a) or (f2 is not None and b < f2 < a):
					blocked_counterclockwise = True
				else:
					blocked_counterclockwise = False
			else:
				if (f1 is not None and (f1 < a or f1 > b)) or (f2 is not None and (f2 < a or f2 > b)):
					blocked_counterclockwise = True
				else:
					blocked_counterclockwise = False
					
		if not blocked_clockwise or not blocked_counterclockwise:
			total_ops += d0
		else:
			if clockwise_dist == d0:
				if a >= b:
					if (f1 is not None and (b < f1 < a)) or (f2 is not None and (b < f2 < a)):
						print(-1)
						return
					else:
						total_ops += M - d0
				else:
					if (f1 is not None and (f1 < a or f1 > b)) or (f2 is not None and (f2 < a or f2 > b)):
						print(-1)
						return
					else:
						total_ops += M - d0
			else:
				if a <= b:
					if (f1 is not None and (a < f1 < b)) or (f2 is not None and (a < f2 < b)):
						print(-1)
						return
					else:
						total_ops += M - d0
				else:
					if (f1 is not None and (f1 > a or f1 < b)) or (f2 is not None and (f2 > a or f2 < b)):
						print(-1)
						return
					else:
						total_ops += M - d0
						
	print(total_ops)

if __name__ == "__main__":
	main()