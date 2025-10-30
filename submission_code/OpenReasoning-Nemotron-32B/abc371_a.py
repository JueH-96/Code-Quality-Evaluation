def main():
	s_ab, s_ac, s_bc = input().split()
	in_degree = {'A': 0, 'B': 0, 'C': 0}
	
	if s_ab == '<':
		in_degree['B'] += 1
	else:
		in_degree['A'] += 1
		
	if s_ac == '<':
		in_degree['C'] += 1
	else:
		in_degree['A'] += 1
		
	if s_bc == '<':
		in_degree['C'] += 1
	else:
		in_degree['B'] += 1
		
	if in_degree['A'] == 1:
		print('A')
	elif in_degree['B'] == 1:
		print('B')
	else:
		print('C')

if __name__ == '__main__':
	main()