def main():
	N = int(input().strip())
	k = N - 1
	if k == 0:
		s_base5 = "0"
	else:
		s_base5 = ""
		num = k
		while num:
			r = num % 5
			num = num // 5
			s_base5 = str(r) + s_base5
	result = ''.join(str(2 * int(d)) for d in s_base5)
	print(result)

if __name__ == '__main__':
	main()