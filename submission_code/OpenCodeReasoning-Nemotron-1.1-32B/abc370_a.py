def main():
	data = input().split()
	L = int(data[0])
	R = int(data[1])
	
	if L + R == 1:
		if L == 1:
			print("Yes")
		else:
			print("No")
	else:
		print("Invalid")

if __name__ == "__main__":
	main()