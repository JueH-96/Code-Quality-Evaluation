def main():
	S = input().strip()
	n = len(S)
	count_upper = sum(1 for c in S if c.isupper())
	if 2 * count_upper > n:
		print(S.upper())
	else:
		print(S.lower())

if __name__ == "__main__":
	main()