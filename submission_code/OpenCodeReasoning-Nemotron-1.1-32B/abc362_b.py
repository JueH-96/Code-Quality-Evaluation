def main():
	points = []
	for _ in range(3):
		data = input().split()
		x = int(data[0])
		y = int(data[1])
		points.append((x, y))
	
	A, B, C = points
	x0, y0 = A
	x1, y1 = B
	x2, y2 = C
	
	dAB = (x1 - x0)**2 + (y1 - y0)**2
	dBC = (x2 - x1)**2 + (y2 - y1)**2
	dAC = (x2 - x0)**2 + (y2 - y0)**2
	
	if dAB + dBC == dAC or dAB + dAC == dBC or dAC + dBC == dAB:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()