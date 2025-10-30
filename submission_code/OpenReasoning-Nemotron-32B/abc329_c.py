def main():
	n = int(input().strip())
	s = input().strip()
	
	if n == 0:
		print(0)
		return
		
	max_run_dict = {}
	current_char = s[0]
	current_run = 1
	
	for i in range(1, n):
		if s[i] == s[i-1]:
			current_run += 1
		else:
			if current_char in max_run_dict:
				if current_run > max_run_dict[current_char]:
					max_run_dict[current_char] = current_run
			else:
				max_run_dict[current_char] = current_run
			current_char = s[i]
			current_run = 1
			
	if current_char in max_run_dict:
		if current_run > max_run_dict[current_char]:
			max_run_dict[current_char] = current_run
	else:
		max_run_dict[current_char] = current_run
		
	total = sum(max_run_dict.values())
	print(total)

if __name__ == "__main__":
	main()