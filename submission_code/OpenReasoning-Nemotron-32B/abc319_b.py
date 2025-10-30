def main():
	n = int(input().strip())
	divisors = [j for j in range(1, 10) if n % j == 0]
	res = []
	for i in range(n + 1):
		found = False
		for j in divisors:
			d = n // j
			if i % d == 0:
				res.append(str(j))
				found = True
				break
		if not found:
			res.append('-')
	print(''.join(res))

if __name__ == '__main__':
	main()