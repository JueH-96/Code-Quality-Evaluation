def main():
	n, x, y, z = map(int, input().split())
	low = min(x, y)
	high = max(x, y)
	if low < z < high:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()