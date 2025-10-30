def main():
	A, M, L, R = map(int, input().split())
	k_min = (L - A + M - 1) // M
	k_max = (R - A) // M
	if k_min > k_max:
		print(0)
	else:
		print(k_max - k_min + 1)

if __name__ == '__main__':
	main()