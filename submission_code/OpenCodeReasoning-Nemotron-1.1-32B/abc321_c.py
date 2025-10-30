import itertools

def main():
	digits = "0123456789"
	numbers = []
	for d in range(1, 11):
		for comb in itertools.combinations(digits, d):
			if max(comb) == '0':
				continue
			num_str = ''.join(sorted(comb, reverse=True))
			num = int(num_str)
			numbers.append(num)
	numbers.sort()
	k = int(input())
	print(numbers[k-1])

if __name__ == '__main__':
	main()