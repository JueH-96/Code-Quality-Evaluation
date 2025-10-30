import itertools

def main():
	K = int(input().strip())
	digits = list(range(10))
	numbers = []
	for d in range(1, 11):
		for comb in itertools.combinations(digits, d):
			if max(comb) == 0:
				continue
			num_str = ''.join(str(x) for x in sorted(comb, reverse=True))
			num_int = int(num_str)
			numbers.append(num_int)
	numbers.sort()
	print(numbers[K-1])

if __name__ == '__main__':
	main()