import math

def main():
	N = int(input().strip())
	d = 1
	while True:
		d3 = d * d * d
		if d3 > N:
			break
		D_val = 3 * d * (4 * N - d3)
		root = math.isqrt(D_val)
		if root * root == D_val:
			numerator = root - 3 * d * d
			if numerator % (6 * d) == 0:
				y = numerator // (6 * d)
				if y > 0:
					x = y + d
					print(f"{x} {y}")
					return
		d += 1
	print(-1)

if __name__ == '__main__':
	main()