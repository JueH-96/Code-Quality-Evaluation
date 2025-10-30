def main():
	n = int(input().strip())
	count = 0
	for _ in range(n):
		s = input().strip()
		if s == "Takahashi":
			count += 1
	print(count)

if __name__ == "__main__":
	main()