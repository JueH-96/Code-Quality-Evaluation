def main():
	import sys
	data = sys.stdin.read().splitlines()
	n, k = map(int, data[0].split())
	s = data[1].strip()
	count = 0
	current = 0
	for char in s:
		if char == 'O':
			current += 1
		else:
			count += current // k
			current = 0
	count += current // k
	print(count)

if __name__ == '__main__':
	main()