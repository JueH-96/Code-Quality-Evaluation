def main():
	ab, ac, bc = input().split()
	less_A = (ab == '<') + (ac == '<')
	less_B = (ab == '>') + (bc == '<')
	less_C = (ac == '>') + (bc == '>')
	if less_A == 1:
		print('A')
	elif less_B == 1:
		print('B')
	elif less_C == 1:
		print('C')

if __name__ == '__main__':
	main()