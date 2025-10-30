import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+2*n]))
	C = list(map(int, data[1+2*n:1+3*n]))
	
	base = 0
	m = 0
	S0 = 0
	S1 = 0
	w_list = []
	
	for i in range(n):
		if A[i] == B[i]:
			if A[i] == 1:
				base += C[i]
		else:
			m += 1
			if A[i] == 0:
				S0 += C[i]
				w_list.append(-C[i])
			else:
				S1 += C[i]
				w_list.append(C[i])
				
	base *= m
	w_list.sort(reverse=True)
	
	term = 0
	for i in range(m):
		term += w_list[i] * (i+1)
		
	ans = base + (m+1) * S0 - S1 + term
	print(ans)

if __name__ == "__main__":
	main()