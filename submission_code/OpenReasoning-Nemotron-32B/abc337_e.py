import sys

def main():
	data = sys.stdin.readline().strip()
	N = int(data)
	
	M = 0
	while (1 << M) < N:
		M += 1
		
	print(M)
	sys.stdout.flush()
	
	for i in range(M):
		bottles = []
		for j in range(1, N+1):
			if ((j - 1) >> i) & 1:
				bottles.append(j)
		line = [str(len(bottles))] + list(map(str, bottles))
		print(" ".join(line))
		sys.stdout.flush()
	
	S = sys.stdin.readline().strip()
	x = 0
	for i in range(M):
		if S[i] == '1':
			x |= (1 << i)
			
	print(x + 1)
	sys.stdout.flush()

if __name__ == "__main__":
	main()