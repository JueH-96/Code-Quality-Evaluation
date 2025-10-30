def main():
	import sys
	data = sys.stdin.read().split()
	L = int(data[0])
	R = int(data[1])
	sequences = []
	current = L
	while current < R:
		if current == 0:
			i0 = 61
		else:
			low_bit = current & -current
			i0 = low_bit.bit_length() - 1
		diff = R - current
		i_max2 = diff.bit_length() - 1
		i_val = min(i0, i_max2)
		block_end = current + (1 << i_val)
		sequences.append((current, block_end))
		current = block_end
	print(len(sequences))
	for pair in sequences:
		print(pair[0], pair[1])

if __name__ == "__main__":
	main()