import sys

def main():
	N = int(input().strip())
	M = 0
	while (1 << M) < N:
		M += 1
		
	print(M)
	
	friends = [[] for _ in range(M)]
	
	for j in range(1, N + 1):
		num = j - 1
		bin_str = bin(num)[2:]
		if len(bin_str) < M:
			bin_str = '0' * (M - len(bin_str)) + bin_str
		for i in range(M):
			if bin_str[i] == '1':
				friends[i].append(j)
				
	for i in range(M):
		k = len(friends[i])
		if k == 0:
			print(0)
		else:
			print(str(k) + " " + " ".join(map(str, friends[i])))
			
	sys.stdout.flush()
	
	S = input().strip()
	x = int(S, 2)
	print(x + 1)

if __name__ == '__main__':
	main()