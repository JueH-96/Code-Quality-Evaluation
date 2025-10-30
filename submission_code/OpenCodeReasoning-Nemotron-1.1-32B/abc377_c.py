import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	N = int(data[0])
	M = int(data[1])
	pieces = []
	idx = 2
	for i in range(M):
		a = int(data[idx])
		b = int(data[idx + 1])
		idx += 2
		pieces.append((a, b))
	
	existing_set = set(pieces)
	
	moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
	attacked_set = set()
	
	for a, b in pieces:
		for dx, dy in moves:
			ni = a + dx
			nj = b + dy
			if 1 <= ni <= N and 1 <= nj <= N:
				attacked_set.add((ni, nj))
	
	attacked_set_empty = attacked_set - existing_set
	total_squares = N * N
	safe = total_squares - M - len(attacked_set_empty)
	print(safe)

if __name__ == "__main__":
	main()