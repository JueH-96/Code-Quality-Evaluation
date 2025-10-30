def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	c = int(data[1])
	times = list(map(int, data[2:2+n]))
	
	count = 0
	last_candy = -10**9
	
	for t in times:
		if t - last_candy >= c:
			count += 1
			last_candy = t
			
	print(count)

if __name__ == "__main__":
	main()