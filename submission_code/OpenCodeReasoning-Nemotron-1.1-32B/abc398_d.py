import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	parts = data[0].split()
	N = int(parts[0])
	R = int(parts[1])
	C = int(parts[2])
	S = data[1].strip()
	
	dx, dy = 0, 0
	seen = set()
	seen.add((0, 0))
	res = []
	
	for char in S:
		if char == 'N':
			dx -= 1
		elif char == 'S':
			dx += 1
		elif char == 'W':
			dy -= 1
		elif char == 'E':
			dy += 1
		
		target = (dx - R, dy - C)
		if target in seen:
			res.append('1')
		else:
			res.append('0')
		
		seen.add((dx, dy))
	
	print(''.join(res))

if __name__ == "__main__":
	main()