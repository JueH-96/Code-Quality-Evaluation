import sys

def main():
	data = sys.stdin.read().split()
	A = int(data[0])
	B = int(data[1])
	M = int(data[2])
	n_val = 2 * (A + B)
	C = [[0] * (n_val + 1) for _ in range(n_val + 1)]
	for i in range(n_val + 1):
		C[i][0] = 1
		if i > 0:
			for j in range(1, i + 1):
				C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % M

	max_k = A + B
	F = [[[0] * (max_k + 1) for _ in range(B + 1)] for __ in range(A + 1)]
	for i in range(A + 1):
		for j in range(B + 1):
			for k in range(max_k + 1):
				if k == 0:
					F[i][j][k] = 1
				else:
					total = 0
					if i >= 1:
						for l in range(0, j):
							if k - i < 0:
								continue
							if j - l - 1 < 0:
								continue
							term = C[k - 1][i - 1]
							term = term * F[i - 1][j][k - i] % M
							term = term * F[1][l][i - 1] % M
							term = term * F[0][j - l - 1][k - i] % M
							total = (total + term) % M
					if j >= 1:
						for l in range(0, i):
							if k - j < 0:
								continue
							if i - l - 1 < 0:
								continue
							term = C[k - 1][j - 1]
							term = term * F[i][j - 1][k - j] % M
							term = term * F[l][1][j - 1] % M
							term = term * F[i - l - 1][0][k - j] % M
							total = (total + term) % M
					F[i][j][k] = total % M

	G = [[[0] * (max_k + 1) for _ in range(B + 1)] for __ in range(A + 1)]
	for i in range(A + 1):
		for j in range(B + 1):
			for k in range(max_k + 1):
				G[i][j][k] = F[i][j][k]

	for i in range(A + 1):
		for j in range(B + 1):
			for k in range(max_k + 1):
				for I in range(0, i):
					for J in range(0, j):
						for K in range(0, k + 1):
							if i - I < 0 or j - J < 0 or k - K < 0:
								continue
							c_val = C[i - I + j - J][i - I]
							term = G[I][J][K] * c_val % M
							term = term * G[i - I][j - J][k - K] % M
							G[i][j][k] = (G[i][j][k] - term) % M

	ans = G[A - 1][B - 1][A + B - 2] % M
	if ans < 0:
		ans += M
	print(ans)

if __name__ == '__main__':
	main()