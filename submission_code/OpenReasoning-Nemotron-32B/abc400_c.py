import math

def main():
	N = int(input().strip())
	part1 = math.isqrt(N // 4)
	part2 = math.isqrt(N // 2)
	total = part1 + part2
	print(total)

if __name__ == '__main__':
	main()