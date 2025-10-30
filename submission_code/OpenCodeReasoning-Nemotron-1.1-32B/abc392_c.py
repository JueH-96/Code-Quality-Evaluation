def main():
	import sys
	input = sys.stdin.read().split()
	n = int(input[0])
	P = list(map(int, input[1:1+n]))
	Q = list(map(int, input[1+n:1+2*n]))
	
	A = [0] * (n + 1)
	for idx in range(n):
		bib_val = Q[idx]
		A[bib_val] = idx
		
	res = []
	for i in range(1, n + 1):
		j = A[i]
		target_idx = P[j] - 1
		res.append(str(Q[target_idx]))
		
	print(" ".join(res))

if __name__ == "__main__":
	main()