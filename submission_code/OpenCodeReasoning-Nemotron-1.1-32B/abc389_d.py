import math

def main():
	R = int(input().strip())
	twoR2 = 2 * R * R
	total = 0
	for i in range(0, R + 1):
		fi = 2 * i * i + 2 * i
		if fi > twoR2 - 1:
			break
		rem = twoR2 - fi - 1
		n = 1 + 2 * rem
		root = math.isqrt(n)
		j0 = (root - 1) // 2
		if i == 0:
			total += 2 * j0 + 1
		else:
			total += 2 * (2 * j0 + 1)
	print(total)

if __name__ == '__main__':
	main()