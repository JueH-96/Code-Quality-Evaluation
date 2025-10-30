import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	A = list(map(int, data[1].split()))
	S = data[2].strip()
	
	left_M0 = [0] * (n+1)
	left_M1 = [0] * (n+1)
	left_M2 = [0] * (n+1)
	
	for i in range(n):
		left_M0[i+1] = left_M0[i]
		left_M1[i+1] = left_M1[i]
		left_M2[i+1] = left_M2[i]
		if S[i] == 'M':
			if A[i] == 0:
				left_M0[i+1] += 1
			elif A[i] == 1:
				left_M1[i+1] += 1
			elif A[i] == 2:
				left_M2[i+1] += 1
				
	right_X0 = [0] * (n+2)
	right_X1 = [0] * (n+2)
	right_X2 = [0] * (n+2)
	right_X0[n] = 0
	right_X1[n] = 0
	right_X2[n] = 0
	
	for i in range(n-1, -1, -1):
		right_X0[i] = right_X0[i+1]
		right_X1[i] = right_X1[i+1]
		right_X2[i] = right_X2[i+1]
		if S[i] == 'X':
			if A[i] == 0:
				right_X0[i] += 1
			elif A[i] == 1:
				right_X1[i] += 1
			elif A[i] == 2:
				right_X2[i] += 1
				
	total_sum = 0
	for j in range(n):
		if S[j] == 'E':
			L0 = left_M0[j]
			L1 = left_M1[j]
			L2 = left_M2[j]
			R0 = right_X0[j+1]
			R1 = right_X1[j+1]
			R2 = right_X2[j+1]
			
			total_j = (L0 + L1 + L2) * (R0 + R1 + R2)
			
			if b == 0:
				no0 = 0
			else:
				no0 = (L1 + L2) * (R1 + R2)
				
			if b == 1:
				no1 = 0
			else:
				no1 = (L0 + L2) * (R0 + R2)
				
			if b == 2:
				no2 = 0
			else:
				no2 = (L0 + L1) * (R0 + R1)
				
			if b == 2:
				no0_no1 = L2 * R2
			else:
				no0_no1 = 0
				
			if b == 1:
				no0_no2 = L1 * R1
			else:
				no0_no2 = 0
				
			if b == 0:
				no1_no2 = L0 * R0
			else:
				no1_no2 = 0
				
			S0_j = total_j - no0
			S1_j = S0_j - (no1 - no0_no1)
			S2_j = total_j - (no0 + no1 + no2) + (no0_no1 + no0_no2 + no1_no2)
			
			total_sum += S0_j + S1_j + S2_j
			
	print(total_sum)

if __name__ == "__main__":
	main()