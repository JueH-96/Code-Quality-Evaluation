def main():
	S = input().strip()
	mapping = {char: idx for idx, char in enumerate(S)}
	seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	total_distance = 0
	current = mapping['A']
	for i in range(1, len(seq)):
		next_char = seq[i]
		next_pos = mapping[next_char]
		total_distance += abs(current - next_pos)
		current = next_pos
	print(total_distance)

if __name__ == "__main__":
	main()