def main():
	import sys
	data = sys.stdin.read().splitlines()
	n, d = map(int, data[0].split())
	s = data[1].strip()
	
	cookies = [i for i, char in enumerate(s) if char == '@']
	remove_set = set(cookies[-d:])
	
	res = []
	for i in range(len(s)):
		if i in remove_set:
			res.append('.')
		else:
			res.append(s[i])
			
	print(''.join(res))

if __name__ == "__main__":
	main()