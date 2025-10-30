def main():
	import sys
	data = sys.stdin.readline().split()
	n = int(data[0])
	s = int(data[1])
	m = int(data[2])
	l = int(data[3])
	
	ans = float('inf')
	max_x = (n + 11) // 12 + 2
	max_y = (n + 7) // 8 + 2
	
	for x in range(0, max_x + 1):
		for y in range(0, max_y + 1):
			total_eggs = 12 * x + 8 * y
			if total_eggs >= n:
				cost = x * l + y * m
				if cost < ans:
					ans = cost
			else:
				rem = n - total_eggs
				z = (rem + 5) // 6
				cost = x * l + y * m + z * s
				if cost < ans:
					ans = cost
					
	print(ans)

if __name__ == "__main__":
	main()