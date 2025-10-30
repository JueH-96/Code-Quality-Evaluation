def main():
	p, q = input().split()
	dist_map = {
		'A': 0,
		'B': 3,
		'C': 4,
		'D': 8,
		'E': 9,
		'F': 14,
		'G': 23
	}
	result = abs(dist_map[p] - dist_map[q])
	print(result)

if __name__ == "__main__":
	main()