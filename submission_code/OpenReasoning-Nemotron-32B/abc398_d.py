import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	first_line = data[0].split()
	N = int(first_line[0])
	R = int(first_line[1])
	C = int(first_line[2])
	S = data[1].strip()
	
	dr = {'N': -1, 'S': 1, 'W': 0, 'E': 0}
	dc = {'N': 0, 'S': 0, 'W': -1, 'E': 1}
	
	dx = [0] * (N + 1)
	dy = [0] * (N + 1)
	
	for i in range(1, N + 1):
		char = S[i - 1]
		dx[i] = dx[i - 1] + dr[char]
		dy[i] = dy[i - 1] + dc[char]
	
	gen_set = set()
	ans = []
	
	for t in range(1, N + 1):
		if (dx[t], dy[t]) == (0, 0):
			gen_t = False
		else:
			if (dx[t], dy[t]) in gen_set:
				gen_t = False
			else:
				gen_t = True
				gen_set.add((dx[t], dy[t]))
		
		if (dx[t], dy[t]) == (R, C):
			ans.append('1')
		else:
			A = (dx[t] - R, dy[t] - C)
			if A in gen_set:
				ans.append('1')
			else:
				ans.append('0')
	
	print(''.join(ans))

if __name__ == "__main__":
	main()