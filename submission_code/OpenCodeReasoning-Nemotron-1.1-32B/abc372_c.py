import sys

def main():
	data = sys.stdin.read().splitlines()
	n, q = map(int, data[0].split())
	s = list(data[1].strip())
	
	total = 0
	for i in range(n - 2):
		if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
			total += 1
			
	out_lines = []
	index = 2
	for _ in range(q):
		parts = data[index].split()
		index += 1
		x = int(parts[0])
		c = parts[1]
		pos = x - 1
		
		if s[pos] == c:
			out_lines.append(str(total))
		else:
			for start in [pos-2, pos-1, pos]:
				if 0 <= start <= n-3:
					if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
						total -= 1
			s[pos] = c
			for start in [pos-2, pos-1, pos]:
				if 0 <= start <= n-3:
					if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
						total += 1
			out_lines.append(str(total))
			
	print("
".join(out_lines))

if __name__ == "__main__":
	main()