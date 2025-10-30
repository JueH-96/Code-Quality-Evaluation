import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("Yes")
		return
	
	first_line = data[0].split()
	N = int(first_line[0])
	M = int(first_line[1])
	H = int(first_line[2])
	K = int(first_line[3])
	S = data[1].strip()
	
	items_set = set()
	for i in range(2, 2 + M):
		x, y = map(int, data[i].split())
		items_set.add((x, y))
	
	current = (0, 0)
	health = H
	
	for move in S:
		x, y = current
		if move == 'R':
			current = (x + 1, y)
		elif move == 'L':
			current = (x - 1, y)
		elif move == 'U':
			current = (x, y + 1)
		elif move == 'D':
			current = (x, y - 1)
		
		health -= 1
		
		if health < 0:
			print("No")
			return
		
		if current in items_set and health < K:
			health = K
			items_set.remove(current)
	
	print("Yes")

if __name__ == "__main__":
	main()