import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
		
	n = int(data[0])
	m = int(data[1])
	pieces = []
	existing_set = set()
	index = 2
	for i in range(m):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		pieces.append((a, b))
		existing_set.add((a, b))
	
	moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
	captured_set = set()
	for (a, b) in pieces:
		for dx, dy in moves:
			x = a + dx
			y = b + dy
			if 1 <= x <= n and 1 <= y <= n:
				captured_set.add((x, y))
				
	unsafe_empty = captured_set - existing_set
	total_safe = n * n - m - len(unsafe_empty)
	print(total_safe)

if __name__ == '__main__':
	main()