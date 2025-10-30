import bisect

def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	total = sum(A)
	B = sorted(A)
	P = [0] * (n + 1)
	for i in range(1, n + 1):
		P[i] = P[i - 1] + B[i - 1]
	
	res = []
	for a in A:
		idx = bisect.bisect_right(B, a)
		s = total - P[idx]
		res.append(str(s))
	
	print(" ".join(res))

if __name__ == "__main__":
	main()