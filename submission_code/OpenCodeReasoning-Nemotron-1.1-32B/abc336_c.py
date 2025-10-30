def main():
	n = int(input().strip())
	k = n - 1
	if k == 0:
		print(0)
	else:
		s = ""
		while k:
			k, r = divmod(k, 5)
			s = str(2 * r) + s
		print(s)

if __name__ == "__main__":
	main()