def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	hands = list(map(int, data[2:2+n]))
	
	count = 0
	remaining = m
	for h in hands:
		if remaining < h:
			remaining = 0
		else:
			remaining -= h
			count += 1
	print(count)

if __name__ == "__main__":
	main()