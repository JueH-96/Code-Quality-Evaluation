MOD = 998244353

import sys

def main():
	data = sys.stdin.read().splitlines()
	t = int(data[0])
	index = 1
	out = []
	for _ in range(t):
		H, W = map(int, data[index].split())
		index += 1
		grid = []
		for i in range(H):
			grid.append(data[index].strip())
			index += 1
		
		valid = True
		for i in range(H):
			count = 0
			for c in grid[i]:
				if c == 'A':
					count += 1
			if count % 2 != 0:
				valid = False
				break
		
		if not valid:
			out.append("0")
			continue
			
		for j in range(W):
			count = 0
			for i in range(H):
				if grid[i][j] == 'A':
					count += 1
			if count % 2 != 0:
				valid = False
				break
				
		if valid:
			out.append("2")
		else:
			out.append("0")
			
	print("
".join(out))

if __name__ == "__main__":
	main()