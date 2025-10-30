def main():
	data = input().split()
	A = list(map(int, data))
	target = [1, 2, 3, 4, 5]
	
	if A == target:
		print("No")
		return
		
	found = False
	for i in range(4):
		B = A.copy()
		B[i], B[i+1] = B[i+1], B[i]
		if B == target:
			found = True
			break
			
	print("Yes" if found else "No")

if __name__ == "__main__":
	main()