def main():
	n = int(input().strip())
	s = input().strip()
	
	total_t = s.count('T')
	total_a = s.count('A')
	
	if total_t > total_a:
		print('T')
	elif total_t < total_a:
		print('A')
	else:
		t_count = 0
		a_count = 0
		t_reach = None
		a_reach = None
		for i, char in enumerate(s):
			if char == 'T':
				t_count += 1
				if t_count == total_t and t_reach is None:
					t_reach = i
			else:
				a_count += 1
				if a_count == total_a and a_reach is None:
					a_reach = i
			if t_reach is not None and a_reach is not None:
				break
				
		if t_reach < a_reach:
			print('T')
		else:
			print('A')

if __name__ == '__main__':
	main()