def main():
	H = int(input().strip())
	d = (H + 1).bit_length()
	print(d)

if __name__ == '__main__':
	main()