def main():
	S = input().strip()
	for i in range(1, 16, 2):
		if S[i] != '0':
			print("No")
			return
	print("Yes")

if __name__ == "__main__":
	main()