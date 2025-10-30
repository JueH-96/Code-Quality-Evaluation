def main():
	s = input().strip()
	insertions = 0
	j = 0
	for c in s:
		if j % 2 == 0:
			exp = 'i'
		else:
			exp = 'o'
		while c != exp:
			insertions += 1
			j += 1
			if j % 2 == 0:
				exp = 'i'
			else:
				exp = 'o'
		j += 1
	if j % 2 == 1:
		insertions += 1
	print(insertions)

if __name__ == "__main__":
	main()