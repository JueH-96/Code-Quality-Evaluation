import sys

def main():
	data = sys.stdin.read().splitlines()
	n, m, H, k = map(int, data[0].split())
	s = data[1].strip()
	items = set()
	for i in range(2, 2 + m):
		x, y = map(int, data[i].split())
		items.add((x, y))
	
	x, y = 0, 0
	health = H
	for move in s:
		if move == 'R':
			x += 1
		elif move == 'L':
			x -= 1
		elif move == 'U':
			y += 1
		elif move == 'D':
			y -= 1
		health -= 1
		if health < 0:
			print("No")
			return
		if (x, y) in items and health < k:
			health = k
	print("Yes")

if __name__ == '__main__':
	main()