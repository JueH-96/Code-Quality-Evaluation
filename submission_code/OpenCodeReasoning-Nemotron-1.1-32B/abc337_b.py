def main():
	S = input().strip()
	state = 0
	valid = True
	for c in S:
		if state == 0:
			if c == 'A':
				continue
			elif c == 'B':
				state = 1
			elif c == 'C':
				state = 2
		elif state == 1:
			if c == 'B':
				continue
			elif c == 'C':
				state = 2
			else:
				valid = False
				break
		elif state == 2:
			if c == 'C':
				continue
			else:
				valid = False
				break
				
	print('Yes' if valid else 'No')

if __name__ == '__main__':
	main()