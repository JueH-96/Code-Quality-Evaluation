def main():
	n = int(input().strip())
	s = input().strip()
	if n == 0:
		print(0)
		return
	max_run = {}
	current_char = s[0]
	current_run = 1
	for i in range(1, n):
		if s[i] == current_char:
			current_run += 1
		else:
			max_run[current_char] = max(max_run.get(current_char, 0), current_run)
			current_char = s[i]
			current_run = 1
	max_run[current_char] = max(max_run.get(current_char, 0), current_run)
	total = sum(max_run.values())
	print(total)

if __name__ == '__main__':
	main()