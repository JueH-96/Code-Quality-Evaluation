def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	a = list(map(int, data[2:2+n]))
	
	starts = 0
	current_empty = k
	i = 0
	
	while i < n:
		if current_empty < a[i]:
			starts += 1
			current_empty = k
		else:
			current_empty -= a[i]
			i += 1
			
	starts += 1
	print(starts)

if __name__ == '__main__':
	main()