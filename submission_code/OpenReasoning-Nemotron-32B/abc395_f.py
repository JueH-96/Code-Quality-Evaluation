import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	X = int(data[1])
	u = []
	d = []
	index = 2
	for i in range(n):
		u_i = int(data[index])
		d_i = int(data[index + 1])
		index += 2
		u.append(u_i)
		d.append(d_i)
	
	H_max0 = min(u_i + d_i for u_i, d_i in zip(u, d))
	total_sum = sum(u) + sum(d)
	
	def feasible(H):
		low = max(0, H - d[0])
		high = u[0]
		if low > high:
			return False
			
		for i in range(1, n):
			L_i = max(0, H - d[i])
			R_i = u[i]
			low_next = max(L_i, low - X)
			high_next = min(R_i, high + X)
			if low_next > high_next:
				return False
			low, high = low_next, high_next
		return True

	lo = 0
	hi = H_max0
	ansH = 0
	while lo <= hi:
		mid = (lo + hi) // 2
		if feasible(mid):
			ansH = mid
			lo = mid + 1
		else:
			hi = mid - 1
			
	result = total_sum - n * ansH
	print(result)

if __name__ == "__main__":
	main()