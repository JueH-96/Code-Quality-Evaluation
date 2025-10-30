import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n, q = map(int, data[0].split())
	heads = [(1, 0)]
	T = 0
	output_lines = []
	
	for i in range(1, q + 1):
		parts = data[i].split()
		if parts[0] == '1':
			direction = parts[1]
			x, y = heads[-1]
			if direction == 'R':
				new_pos = (x + 1, y)
			elif direction == 'L':
				new_pos = (x - 1, y)
			elif direction == 'U':
				new_pos = (x, y + 1)
			elif direction == 'D':
				new_pos = (x, y - 1)
			heads.append(new_pos)
			T += 1
		else:
			p = int(parts[1])
			if T >= p - 1:
				pos = heads[T - (p - 1)]
				output_lines.append(f"{pos[0]} {pos[1]}")
			else:
				x0 = p - T
				output_lines.append(f"{x0} 0")
	
	for line in output_lines:
		print(line)

if __name__ == '__main__':
	main()