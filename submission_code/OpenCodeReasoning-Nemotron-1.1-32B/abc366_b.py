import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	N = int(data[0])
	S = []
	for i in range(1, 1 + N):
		S.append(data[i].strip())
	
	M = max(len(s) for s in S) if S else 0
	
	if M == 0:
		for _ in range(N):
			print("")
		return
	
	grid = [[None] * N for _ in range(M)]
	
	for i in range(N):
		s = S[i]
		k0 = N - i - 1
		for j in range(len(s)):
			grid[j][k0] = s[j]
	
	for j in range(M):
		max_row = -1
		for k in range(N):
			if grid[j][k] is not None:
				if k > max_row:
					max_row = k
		
		if max_row == -1:
			print("")
		else:
			arr = []
			for k in range(max_row + 1):
				if grid[j][k] is not None:
					arr.append(grid[j][k])
				else:
					arr.append('*')
			while arr and arr[-1] == '*':
				arr.pop()
			print(''.join(arr))

if __name__ == '__main__':
	main()