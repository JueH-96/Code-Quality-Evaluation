def main():
	n = int(input().strip())
	rects = []
	for _ in range(n):
		a, b, c, d = map(int, input().split())
		rects.append((a, b, c, d))
	
	total_area = 0
	for x in range(100):
		for y in range(100):
			for a, b, c, d in rects:
				if a <= x and b >= x + 1 and c <= y and d >= y + 1:
					total_area += 1
					break
	print(total_area)

if __name__ == "__main__":
	main()