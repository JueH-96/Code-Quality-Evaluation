import itertools

def nCr(n, r):
	if r < 0 or r > n:
		return 0
	r = min(r, n - r)
	num = 1
	for i in range(1, r + 1):
		num = num * (n - i + 1) // i
	return num

def main():
	N, M = map(int, input().split())
	L = M - 10 * (N - 1)
	total = nCr(L + N - 1, N)
	print(total)
	for seq in itertools.combinations_with_replacement(range(1, L + 1), N):
		A = [str(seq[i] + 10 * i) for i in range(N)]
		print(" ".join(A))

if __name__ == "__main__":
	main()