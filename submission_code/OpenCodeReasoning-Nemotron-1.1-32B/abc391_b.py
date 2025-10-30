def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n_m = data[0].split()
	N = int(n_m[0])
	M = int(n_m[1])
	
	S = []
	for i in range(1, 1 + N):
		S.append(data[i].strip())
	
	T = []
	for i in range(1 + N, 1 + N + M):
		T.append(data[i].strip())
	
	for a in range(0, N - M + 1):
		for b in range(0, N - M + 1):
			match = True
			for i in range(0, M):
				for j in range(0, M):
					if S[a + i][b + j] != T[i][j]:
						match = False
						break
				if not match:
					break
			if match:
				print(f"{a + 1} {b + 1}")
				return

if __name__ == "__main__":
	main()