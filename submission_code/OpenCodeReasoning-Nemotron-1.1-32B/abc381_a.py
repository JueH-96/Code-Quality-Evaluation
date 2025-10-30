def main():
	n = int(input().strip())
	s = input().strip()
	
	if n % 2 == 0:
		print("No")
	else:
		mid = n // 2
		expected = '1' * mid + '/' + '2' * mid
		if s == expected:
			print("Yes")
		else:
			print("No")

if __name__ == "__main__":
	main()