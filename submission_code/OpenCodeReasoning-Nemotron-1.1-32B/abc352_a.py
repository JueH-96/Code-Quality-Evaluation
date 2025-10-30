def main():
	data = input().split()
	N = int(data[0])
	X = int(data[1])
	Y = int(data[2])
	Z = int(data[3])
	low = min(X, Y)
	high = max(X, Y)
	if low < Z < high:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()