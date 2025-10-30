import math

def main():
	A, B, C, D = map(int, input().split())
	area = (C - A) * (D - B)
	x0 = A + 0.5
	y0 = B + 0.5
	a = math.floor(x0)
	b = math.floor(y0 / 2)
	c = math.floor((x0 + y0 - 1) / 2)
	if (a + b + c) % 2 == 0:
		area += 1
	print(area)

if __name__ == '__main__':
	main()