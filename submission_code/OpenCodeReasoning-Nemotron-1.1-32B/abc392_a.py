def main():
	data = input().split()
	a, b, c = map(int, data)
	if a * b == c or a * c == b or b * c == a:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()