def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	hands = list(map(int, data[2:2+n]))
	
	rem = m
	count = 0
	for h in hands:
		if rem >= h:
			count += 1
			rem -= h
		else:
			rem = 0
	print(count)

if __name__ == "__main__":
	main()