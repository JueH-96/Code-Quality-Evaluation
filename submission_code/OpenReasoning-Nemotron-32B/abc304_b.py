def main():
	n = int(input().strip())
	s = str(n)
	d = len(s)
	if d <= 3:
		print(n)
	else:
		base = 10 ** (d - 3)
		result = (n // base) * base
		print(result)

if __name__ == "__main__":
	main()