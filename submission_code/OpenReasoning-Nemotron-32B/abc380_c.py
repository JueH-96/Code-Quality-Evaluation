def main():
	import sys
	data = sys.stdin.read().splitlines()
	n, k = map(int, data[0].split())
	s = data[1].strip()
	
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
			
	prev_block = blocks[k-2]
	move_block = blocks[k-1]
	
	r_prev0 = prev_block[1]
	l_move0 = move_block[0]
	r_move0 = move_block[1]
	L = r_move0 - l_move0 + 1
	
	parts = []
	parts.append(s[:r_prev0+1])
	parts.append('1' * L)
	start_zero = r_prev0 + 1 + L
	if start_zero <= r_move0:
		zeros_len = r_move0 - start_zero + 1
		parts.append('0' * zeros_len)
	parts.append(s[r_move0+1:])
	
	result = ''.join(parts)
	print(result)

if __name__ == "__main__":
	main()