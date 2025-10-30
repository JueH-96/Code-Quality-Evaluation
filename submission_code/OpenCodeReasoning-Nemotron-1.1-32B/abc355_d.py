import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	events = []
	index = 1
	for i in range(n):
		l = int(data[index])
		r = int(data[index + 1])
		index += 2
		events.append((l, 0))
		events.append((r, 1))
	
	events.sort(key=lambda x: (x[0], x[1]))
	
	open_count = 0
	ans = 0
	for coord, typ in events:
		if typ == 0:
			ans += open_count
			open_count += 1
		else:
			open_count -= 1
			
	print(ans)

if __name__ == "__main__":
	main()