def main():
	import sys
	data = sys.stdin.read().splitlines()
	n, m = map(int, data[0].split())
	S = data[1].strip()
	T = data[2].strip()
	
	prefix = (T[:n] == S)
	suffix = (T[m-n:] == S)
	
	if prefix and suffix:
		print(0)
	elif prefix:
		print(1)
	elif suffix:
		print(2)
	else:
		print(3)

if __name__ == '__main__':
	main()