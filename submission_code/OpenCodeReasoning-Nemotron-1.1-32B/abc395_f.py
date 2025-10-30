import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	X = int(data[1])
	U = []
	D = []
	index = 2
	for i in range(n):
		u = int(data[index])
		d = int(data[index + 1])
		index += 2
		U.append(u)
		D.append(d)
	
	total_initial = sum(U) + sum(D)
	min_total = min(U[i] + D[i] for i in range(n))
	
	def feasible(h):
		low_cur = max(0, h - D[0])
		high_cur = min(U[0], h)
		if low_cur > high_cur:
			return False
			
		for i in range(1, n):
			low_bound = max(0, h - D[i])
			high_bound = min(U[i], h)
			if low_bound > high_bound:
				return False
				
			low_next = max(low_bound, low_cur - X)
			high_next = min(high_bound, high_cur + X)
			if low_next > high_next:
				return False
				
			low_cur, high_cur = low_next, high_next
			
		return True
		
	lo, hi = 0, min_total
	best_h = 0
	while lo <= hi:
		mid = (lo + hi) // 2
		if feasible(mid):
			best_h = mid
			lo = mid + 1
		else:
			hi = mid - 1
			
	cost = total_initial - n * best_h
	print(cost)

if __name__ == '__main__':
	main()