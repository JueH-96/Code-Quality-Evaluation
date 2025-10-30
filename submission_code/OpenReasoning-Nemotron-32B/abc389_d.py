import math

def main():
	R = int(input().strip())
	R2 = R * R
	limit = R2 - 1
	total = 0
	i = 0
	while True:
		K = limit - (i * i + i)
		if K < 0:
			break
		disc = 1 + 4 * K
		s = math.isqrt(disc)
		j_max = (s - 1) // 2
		if i == 0:
			total += 1 + 2 * j_max
		else:
			total += 2 + 4 * j_max
		i += 1
	print(total)

if __name__ == "__main__":
	main()