def main():
	data = input().split()
	a, b, c, d, e, f = map(int, data)
	data2 = input().split()
	g, h, i, j, k, l = map(int, data2)
	
	x0 = max(a, g)
	x1 = min(d, j)
	y0 = max(b, h)
	y1 = min(e, k)
	z0 = max(c, i)
	z1 = min(f, l)
	
	if x0 < x1 and y0 < y1 and z0 < z1:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()