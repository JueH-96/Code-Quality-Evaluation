def main():
	n, t, a = map(int, input().split())
	if 2 * t > n or 2 * a > n:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()