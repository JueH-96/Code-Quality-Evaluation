def main():
	total_sum = 2025
	X = int(input().strip())
	count = 0
	for i in range(1, 10):
		if X % i == 0:
			j = X // i
			if 1 <= j <= 9:
				count += 1
	result = total_sum - count * X
	print(result)

if __name__ == '__main__':
	main()