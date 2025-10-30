def main():
	import sys
	data = sys.stdin.read().splitlines()
	n, d = map(int, data[0].split())
	s = data[1].strip()
	
	positions = []
	for i, char in enumerate(s):
		if char == '@':
			positions.append(i)
			
	k = len(positions)
	remaining_indices = set(positions[:k - d])
	
	res = []
	for i in range(n):
		if i in remaining_indices:
			res.append('@')
		else:
			res.append('.')
			
	print(''.join(res))

if __name__ == '__main__':
	main()