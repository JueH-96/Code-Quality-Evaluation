def main():
	n = int(input().strip())
	D_list = list(map(int, input().split()))
	
	total = 0
	for month in range(1, n + 1):
		s = str(month)
		if all(c == s[0] for c in s):
			d_int = int(s[0])
			if d_int <= D_list[month - 1]:
				total += 1
			two_digit = 11 * d_int
			if two_digit <= D_list[month - 1]:
				total += 1
	print(total)

if __name__ == "__main__":
	main()