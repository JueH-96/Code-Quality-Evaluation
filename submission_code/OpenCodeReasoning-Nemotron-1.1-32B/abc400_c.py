import math

def main():
	N = int(input().strip())
	total = 0
	base = 2
	while base <= N:
		X = N // base
		r = math.isqrt(X)
		total += (r + 1) // 2
		base *= 2
	print(total)

if __name__ == '__main__':
	main()