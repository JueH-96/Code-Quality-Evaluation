def main():
	N = input().strip()
	if len(N) == 1:
		print("Yes")
		return
	for i in range(len(N) - 1):
		if N[i] <= N[i+1]:
			print("No")
			return
	print("Yes")

if __name__ == "__main__":
	main()