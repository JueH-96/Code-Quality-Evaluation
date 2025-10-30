def main():
	N = int(input().strip())
	r = N % 5
	lower = N - r
	upper = lower + 5
	if upper > 100:
		print(lower)
	else:
		if r <= 2:
			print(lower)
		else:
			print(upper)

if __name__ == '__main__':
	main()