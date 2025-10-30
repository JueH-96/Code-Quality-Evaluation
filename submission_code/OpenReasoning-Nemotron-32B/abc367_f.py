import sys
import random

MOD1 = 10**9 + 7
MOD2 = 10**9 + 9

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	q = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+2*n]))
	queries = []
	idx = 2 + 2 * n
	for i in range(q):
		l = int(data[idx])
		r = int(data[idx+1])
		L = int(data[idx+2])
		R = int(data[idx+3])
		idx += 4
		queries.append((l, r, L, R))
	
	rand1 = [0] * (n+1)
	rand2 = [0] * (n+1)
	
	for i in range(1, n+1):
		rand1[i] = random.randint(0, MOD1 - 1)
		rand2[i] = random.randint(0, MOD2 - 1)
	
	P1A = [0] * (n+1)
	P2A = [0] * (n+1)
	for i in range(1, n+1):
		num = A[i-1]
		P1A[i] = (P1A[i-1] + rand1[num]) % MOD1
		P2A[i] = (P2A[i-1] + rand2[num]) % MOD2
		
	P1B = [0] * (n+1)
	P2B = [0] * (n+1)
	for i in range(1, n+1):
		num = B[i-1]
		P1B[i] = (P1B[i-1] + rand1[num]) % MOD1
		P2B[i] = (P2B[i-1] + rand2[num]) % MOD2
		
	out_lines = []
	for (l, r, L, R) in queries:
		totalA1 = (P1A[r] - P1A[l-1]) % MOD1
		totalA2 = (P2A[r] - P2A[l-1]) % MOD2
		totalB1 = (P1B[R] - P1B[L-1]) % MOD1
		totalB2 = (P2B[R] - P2B[L-1]) % MOD2
		
		if totalA1 == totalB1 and totalA2 == totalB2:
			out_lines.append("Yes")
		else:
			out_lines.append("No")
			
	sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
	main()