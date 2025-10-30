import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	n, m = map(int, data[0].split())
	s = data[1].strip()
	t = data[2].strip()
	
	current = [int(char) for char in s]
	queues = [deque() for _ in range(9)]
	
	for i, digit in enumerate(current):
		queues[digit - 1].append(i)
	
	min_digit = 1
	for k in range(m):
		d = int(t[k])
		while min_digit <= 9 and not queues[min_digit - 1]:
			min_digit += 1
		if min_digit > 9:
			i = n - 1
		else:
			i = queues[min_digit - 1].pop()
		
		if min_digit <= 9:
			queues[min_digit - 1].append(i)
		queues[d - 1].append(i)
		current[i] = d

	print(''.join(str(x) for x in current))

if __name__ == '__main__':
	main()