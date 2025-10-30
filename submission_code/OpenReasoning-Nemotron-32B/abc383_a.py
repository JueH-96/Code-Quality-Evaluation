def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	events = []
	for i in range(1, n + 1):
		parts = data[i].split()
		t = int(parts[0])
		v = int(parts[1])
		events.append((t, v))
	
	water = 0
	prev_time = 0
	for t, v in events:
		gap = t - prev_time
		water = max(0, water - gap)
		water += v
		prev_time = t
	
	print(water)

if __name__ == '__main__':
	main()