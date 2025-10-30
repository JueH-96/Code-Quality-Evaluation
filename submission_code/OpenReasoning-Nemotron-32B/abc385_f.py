import collections

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	X = []
	H = []
	index = 1
	for i in range(n):
		x = int(data[index])
		h = int(data[index+1])
		index += 2
		X.append(x)
		H.append(h)
	
	if n == 1:
		print(-1)
		return
		
	upper = []
	lower = []
	M = -10**18
	
	for i in range(n):
		x_i = X[i]
		h_i = H[i]
		best_slope = 10**18
		
		if upper:
			lo = 0
			hi = len(upper) - 1
			while hi - lo > 2:
				m1 = lo + (hi - lo) // 3
				m2 = hi - (hi - lo) // 3
				x1, y1 = upper[m1]
				x2, y2 = upper[m2]
				if x_i == x1 or x_i == x2:
					break
				s1 = (h_i - y1) / (x_i - x1)
				s2 = (h_i - y2) / (x_i - x2)
				if s1 < s2:
					hi = m2
				else:
					lo = m1
			for idx in range(lo, hi + 1):
				xj, yj = upper[idx]
				if x_i == xj:
					continue
				slope_val = (h_i - yj) / (x_i - xj)
				if slope_val < best_slope:
					best_slope = slope_val
					
		if lower:
			lo = 0
			hi = len(lower) - 1
			while hi - lo > 2:
				m1 = lo + (hi - lo) // 3
				m2 = hi - (hi - lo) // 3
				x1, y1 = lower[m1]
				x2, y2 = lower[m2]
				if x_i == x1 or x_i == x2:
					break
				s1 = (h_i - y1) / (x_i - x1)
				s2 = (h_i - y2) / (x_i - x2)
				if s1 < s2:
					hi = m2
				else:
					lo = m1
			for idx in range(lo, hi + 1):
				xj, yj = lower[idx]
				if x_i == xj:
					continue
				slope_val = (h_i - yj) / (x_i - xj)
				if slope_val < best_slope:
					best_slope = slope_val
					
		if best_slope < 10**18:
			candidate_i = h_i - x_i * best_slope
			if candidate_i > M:
				M = candidate_i
				
		while len(upper) >= 2:
			a = upper[-2]
			b = upper[-1]
			cross = (b[0] - a[0]) * (h_i - b[1]) - (b[1] - a[1]) * (x_i - b[0])
			if cross >= 0:
				upper.pop()
			else:
				break
		upper.append((x_i, h_i))
		
		while len(lower) >= 2:
			a = lower[-2]
			b = lower[-1]
			cross = (b[0] - a[0]) * (h_i - b[1]) - (b[1] - a[1]) * (x_i - b[0])
			if cross <= 0:
				lower.pop()
			else:
				break
		lower.append((x_i, h_i))
		
	if M < 0:
		print(-1)
	else:
		print("{:.15f}".format(M))

if __name__ == '__main__':
	main()