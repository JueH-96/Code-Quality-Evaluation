import math

def main():
	N = int(input().strip())
	d_max = 1
	while d_max * d_max * d_max < 4 * N:
		d_max += 1
	d_max += 100

	for d in range(1, d_max + 1):
		if N % d != 0:
			continue
		quotient = N // d
		D = 12 * quotient - 3 * d * d
		if D < 0:
			continue
		root = math.isqrt(D)
		if root * root != D:
			continue
		numerator = -3 * d + root
		if numerator <= 0:
			continue
		if numerator % 6 != 0:
			continue
		y = numerator // 6
		x = y + d
		print(f"{x} {y}")
		return

	print(-1)

if __name__ == '__main__':
	main()