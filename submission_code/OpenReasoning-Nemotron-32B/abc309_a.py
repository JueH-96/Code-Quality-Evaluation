def main():
	A, B = map(int, input().split())
	rowA = (A - 1) // 3
	rowB = (B - 1) // 3
	if rowA != rowB:
		print("No")
	else:
		colA = (A - 1) % 3
		colB = (B - 1) % 3
		if abs(colA - colB) == 1:
			print("Yes")
		else:
			print("No")

if __name__ == "__main__":
	main()