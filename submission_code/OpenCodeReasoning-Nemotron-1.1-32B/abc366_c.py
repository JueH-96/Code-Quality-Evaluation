import sys

def main():
	data = sys.stdin.read().splitlines()
	Q = int(data[0].strip())
	distinct = 0
	freq = {}
	output_lines = []
	index = 1
	for i in range(Q):
		parts = data[index].split()
		index += 1
		if parts[0] == '1':
			x = int(parts[1])
			if x not in freq or freq[x] == 0:
				distinct += 1
			freq[x] = freq.get(x, 0) + 1
		elif parts[0] == '2':
			x = int(parts[1])
			freq[x] -= 1
			if freq[x] == 0:
				distinct -= 1
		elif parts[0] == '3':
			output_lines.append(str(distinct))
	
	print("
".join(output_lines))

if __name__ == "__main__":
	main()