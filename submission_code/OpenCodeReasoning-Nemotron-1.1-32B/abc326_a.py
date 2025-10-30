def main():
	X, Y = map(int, input().split())
	if (1 <= Y - X <= 2) or (1 <= X - Y <= 3):
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()