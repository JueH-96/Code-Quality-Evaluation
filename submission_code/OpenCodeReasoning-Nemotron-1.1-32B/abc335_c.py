import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, q = map(int, data[0].split())
	head_pos = [(1, 0)]
	t = 0
	output_lines = []
	index = 1
	for _ in range(q):
		parts = data[index].split()
		index += 1
		if parts[0] == '1':
			c = parts[1]
			x, y = head_pos[-1]
			if c == 'R':
				x += 1
			elif c == 'L':
				x -= 1
			elif c == 'U':
				y += 1
			elif c == 'D':
				y -= 1
			head_pos.append((x, y))
			t += 1
		else:
			p = int(parts[1])
			if t < p - 1:
				x = p - t
				y = 0
				output_lines.append(f"{x} {y}")
			else:
				idx = t - (p - 1)
				x, y = head_pos[idx]
				output_lines.append(f"{x} {y}")
	for line in output_lines:
		print(line)

if __name__ == '__main__':
	main()