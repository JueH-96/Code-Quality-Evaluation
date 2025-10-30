def main():
	data = []
	for _ in range(2):
		line = input().split()
		data.append([int(x) for x in line])
	
	a, b, c, d, e, f = data[0]
	g, h, i, j, k, l = data[1]
	
	x_low = max(a, g)
	x_high = min(d, j)
	y_low = max(b, h)
	y_high = min(e, k)
	z_low = max(c, i)
	z_high = min(f, l)
	
	if x_low < x_high and y_low < y_high and z_low < z_high:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()