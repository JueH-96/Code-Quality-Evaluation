def main():
	s = input().strip()
	inserted = 0
	pos = 0
	for c in s:
		while True:
			if pos % 2 == 0:
				expected = 'i'
			else:
				expected = 'o'
			if c == expected:
				pos += 1
				break
			else:
				inserted += 1
				pos += 1
	if pos % 2 == 1:
		inserted += 1
	print(inserted)

if __name__ == '__main__':
	main()