def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, k = map(int, data[0].split())
	s = data[1].strip()
	
	if n == 0:
		print("")
		return
		
	blocks = []
	i = 0
	n_val = len(s)
	while i < n_val:
		if s[i] == '1':
			start = i
			j = i
			while j < n_val and s[j] == '1':
				j += 1
			end = j - 1
			blocks.append((start, end))
			i = j
		else:
			i += 1
			
	if k > len(blocks):
		print(s)
		return
		
	block_prev = blocks[k-2]
	block_k = blocks[k-1]
	
	part1 = s[:block_prev[1]+1]
	L = block_k[1] - block_k[0] + 1
	part2 = '1' * L
	zeros_length = block_k[0] - block_prev[1] - 1
	part3 = '0' * zeros_length
	part4 = s[block_k[1]+1:]
	
	result = part1 + part2 + part3 + part4
	print(result)

if __name__ == '__main__':
	main()