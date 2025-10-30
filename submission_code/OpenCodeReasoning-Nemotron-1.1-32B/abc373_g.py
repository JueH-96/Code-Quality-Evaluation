def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	index = 1
	P = []
	for i in range(1, n+1):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		P.append((a, b, i))
	
	Q = []
	for i in range(1, n+1):
		c = int(data[index])
		d = int(data[index+1])
		index += 2
		Q.append((c, d, i))
	
	available_P = P[:]
	available_Q = Q[:]
	res = [0] * (n+1)
	
	for _ in range(n):
		minP = None
		for p in available_P:
			if minP is None:
				minP = p
			else:
				if p[0] < minP[0] or (p[0] == minP[0] and p[1] < minP[1]):
					minP = p
		minQ = None
		for q in available_Q:
			if minQ is None:
				minQ = q
			else:
				if q[0] < minQ[0] or (q[0] == minQ[0] and q[1] < minQ[1]):
					minQ = q
		
		if minP is None and minQ is None:
			break
			
		if minP is None:
			min_point = minQ
			min_type = 'Q'
		elif minQ is None:
			min_point = minP
			min_type = 'P'
		else:
			if minP[0] < minQ[0] or (minP[0] == minQ[0] and minP[1] < minQ[1]):
				min_point = minP
				min_type = 'P'
			else:
				min_point = minQ
				min_type = 'Q'
				
		if min_type == 'P':
			x0, y0, idx_p = min_point
			available_P = [p for p in available_P if p[2] != idx_p]
			candidate = None
			for q in available_Q:
				if candidate is None:
					candidate = q
				else:
					dx1 = candidate[0] - x0
					dy1 = candidate[1] - y0
					dx2 = q[0] - x0
					dy2 = q[1] - y0
					cross = dx1 * dy2 - dx2 * dy1
					if cross > 0:
						candidate = q
			res[idx_p] = candidate[2]
			available_Q = [q for q in available_Q if q[2] != candidate[2]]
		else:
			x0, y0, idx_q = min_point
			available_Q = [q for q in available_Q if q[2] != idx_q]
			candidate = None
			for p in available_P:
				if candidate is None:
					candidate = p
				else:
					dx1 = candidate[0] - x0
					dy1 = candidate[1] - y0
					dx2 = p[0] - x0
					dy2 = p[1] - y0
					cross = dx1 * dy2 - dx2 * dy1
					if cross > 0:
						candidate = p
			res[candidate[2]] = idx_q
			available_P = [p for p in available_P if p[2] != candidate[2]]
			
	out_list = []
	for i in range(1, n+1):
		out_list.append(str(res[i]))
	print(" ".join(out_list))

if __name__ == '__main__':
	main()