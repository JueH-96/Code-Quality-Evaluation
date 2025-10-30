def main():
	s = input().strip()
	from collections import defaultdict
	char_positions = defaultdict(list)
	for index, char in enumerate(s):
		char_positions[char].append(index)
	
	total_ans = 0
	for char, positions in char_positions.items():
		m = len(positions)
		if m < 2:
			continue
		s_val = sum(positions)
		s_j = 0
		for j, pos in enumerate(positions):
			s_j += pos * j
		term = 2 * s_j - (m - 1) * s_val
		num_pairs = m * (m - 1) // 2
		total_char = term - num_pairs
		total_ans += total_char
		
	print(total_ans)

if __name__ == "__main__":
	main()