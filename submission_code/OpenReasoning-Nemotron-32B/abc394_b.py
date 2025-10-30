def main():
	n = int(input().strip())
	strings = []
	for _ in range(n):
		s = input().strip()
		strings.append(s)
	
	strings.sort(key=len)
	
	result = ''.join(strings)
	print(result)

if __name__ == '__main__':
	main()