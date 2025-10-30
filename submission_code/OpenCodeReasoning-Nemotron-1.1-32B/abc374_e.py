import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	X = int(data[1])
	A = []
	P = []
	B = []
	Q = []
	index = 2
	for i in range(n):
		a = int(data[index]); p = int(data[index+1]); b = int(data[index+2]); q = int(data[index+3])
		index += 4
		A.append(a)
		P.append(p)
		B.append(b)
		Q.append(q)
	
	low = 0
	high = 10**18
	
	while low <= high:
		mid = (low + high) // 2
		total_cost = 0
		valid = True
		for i in range(n):
			a = A[i]
			p = P[i]
			b = B[i]
			q = Q[i]
			if mid == 0:
				cost_i = 0
			elif a == 0 and b == 0:
				if mid > 0:
					valid = False
				else:
					cost_i = 0
			elif a == 0:
				t_i = (mid + b - 1) // b
				cost_i = t_i * q
			elif b == 0:
				s_i = (mid + a - 1) // a
				cost_i = s_i * p
			else:
				cost1 = ((mid + a - 1) // a) * p
				cost2 = ((mid + b - 1) // b) * q
				cost_i = min(cost1, cost2)
				
				s_i_max = min((mid + a - 1) // a, b - 1)
				for s_i in range(0, s_i_max + 1):
					rem = mid - a * s_i
					if rem <= 0:
						t_i = 0
					else:
						t_i = (rem + b - 1) // b
					cost = s_i * p + t_i * q
					if cost < cost_i:
						cost_i = cost
				
				t_i_max = min((mid + b - 1) // b, a - 1)
				for t_i in range(0, t_i_max + 1):
					rem = mid - b * t_i
					if rem <= 0:
						s_i = 0
					else:
						s_i = (rem + a - 1) // a
					cost = s_i * p + t_i * q
					if cost < cost_i:
						cost_i = cost
			total_cost += cost_i
			if total_cost > X:
				break
				
		if not valid or total_cost > X:
			high = mid - 1
		else:
			low = mid + 1
			
	print(high)

if __name__ == '__main__':
	main()