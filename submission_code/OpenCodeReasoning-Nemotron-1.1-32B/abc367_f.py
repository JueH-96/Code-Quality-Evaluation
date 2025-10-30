import sys
import random

mod1 = 10**9 + 7
mod2 = 10**9 + 9

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	A_arr = [int(next(it)) for _ in range(n)]
	B_arr = [int(next(it)) for _ in range(n)]
	
	rand1 = [0] * (n + 1)
	rand2 = [0] * (n + 1)
	for i in range(1, n + 1):
		rand1[i] = random.randint(0, mod1 - 1)
		rand2[i] = random.randint(0, mod2 - 1)
	
	P1_A = [0] * (n + 1)
	P2_A = [0] * (n + 1)
	for i in range(1, n + 1):
		x = A_arr[i - 1]
		P1_A[i] = (P1_A[i - 1] + rand1[x]) % mod1
		P2_A[i] = (P2_A[i - 1] + rand2[x]) % mod2
		
	P1_B = [0] * (n + 1)
	P2_B = [0] * (n + 1)
	for i in range(1, n + 1):
		x = B_arr[i - 1]
		P1_B[i] = (P1_B[i - 1] + rand1[x]) % mod1
		P2_B[i] = (P2_B[i - 1] + rand2[x]) % mod2
		
	out_lines = []
	for _ in range(q):
		l = int(next(it))
		r = int(next(it))
		L = int(next(it))
		R = int(next(it))
		lenA = r - l + 1
		lenB = R - L + 1
		if lenA != lenB:
			out_lines.append("No")
		else:
			s1_A = (P1_A[r] - P1_A[l - 1]) % mod1
			s2_A = (P2_A[r] - P2_A[l - 1]) % mod2
			s1_B = (P1_B[R] - P1_B[L - 1]) % mod1
			s2_B = (P2_B[R] - P2_B[L - 1]) % mod2
			if s1_A == s1_B and s2_A == s2_B:
				out_lines.append("Yes")
			else:
				out_lines.append("No")
				
	print("
".join(out_lines))

if __name__ == '__main__':
	main()