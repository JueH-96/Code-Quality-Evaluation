def main():
	n = int(input().strip())
	s = input().strip()
	
	res = []
	stack = []
	
	for c in s:
		if c == '(':
			res.append(c)
			stack.append(len(res) - 1)
		elif c == ')':
			if stack:
				j = stack.pop()
				del res[j:]
			else:
				res.append(c)
		else:
			res.append(c)
			
	print(''.join(res))

if __name__ == "__main__":
	main()