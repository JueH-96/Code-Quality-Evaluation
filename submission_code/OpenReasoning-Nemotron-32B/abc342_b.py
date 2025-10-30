def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	
	pos = [0] * (n + 1)
	for index, person in enumerate(arr):
		pos[person] = index
		
	output_lines = []
	index_ptr = 1 + n + 1
	for _ in range(q):
		a = int(data[index_ptr])
		b = int(data[index_ptr + 1])
		index_ptr += 2
		if pos[a] < pos[b]:
			output_lines.append(str(a))
		else:
			output_lines.append(str(b))
			
	print("
".join(output_lines))

if __name__ == "__main__":
	main()