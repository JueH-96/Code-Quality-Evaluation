import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	M_val = int(data[1])
	P_list = list(map(int, data[2:2+n]))
	
	if M_val == 0:
		print(0)
		return
		
	distinct = {}
	for p in P_list:
		distinct[p] = distinct.get(p, 0) + 1
		
	min_p = min(P_list)
	T_max_upper = M_val // min_p
	
	low_T = 0
	high_T = T_max_upper
	while low_T <= high_T:
		mid_T = (low_T + high_T) // 2
		if mid_T == 0:
			low_T = mid_T + 1
			continue
			
		low_x = 0
		high_x = 10**30
		for _ in range(90):
			mid_x = (low_x + high_x) // 2
			F_mid = 0
			for p, cnt in distinct.items():
				k = (mid_x + p) // (2 * p)
				F_mid += cnt * k
			if F_mid >= mid_T:
				high_x = mid_x
			else:
				low_x = mid_x + 1
				
		x0 = low_x
		
		if x0 == 0:
			F1 = 0
			G1 = 0
		else:
			F1 = 0
			G1 = 0
			overM = False
			for p, cnt in distinct.items():
				k = (x0 - 1 + p) // (2 * p)
				F1 += cnt * k
				term = cnt * (k * k * p)
				G1 += term
				if G1 > M_val:
					overM = True
					break
			if overM:
				S_mid = M_val + 1
			else:
				S_mid = G1 + x0 * (mid_T - F1)
				
		if S_mid <= M_val:
			low_T = mid_T + 1
		else:
			high_T = mid_T - 1
			
	print(high_T)

if __name__ == '__main__':
	main()