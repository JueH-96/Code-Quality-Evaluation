def main():
	n = int(input().strip())
	D = list(map(int, input().split()))
	
	count = 0
	for i in range(1, n + 1):
		s_i = str(i)
		if len(set(s_i)) != 1:
			continue
		digit_i = s_i[0]
		days_in_month = D[i - 1]
		for j in range(1, days_in_month + 1):
			s_j = str(j)
			if len(set(s_j)) == 1 and s_j[0] == digit_i:
				count += 1
	print(count)

if __name__ == "__main__":
	main()