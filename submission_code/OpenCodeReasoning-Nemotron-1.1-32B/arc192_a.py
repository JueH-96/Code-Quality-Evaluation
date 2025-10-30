def main():
	n = int(input().strip())
	a = list(map(int, input().split()))
	
	if all(x == 0 for x in a):
		print("Yes" if n % 2 == 0 else "No")
		return
		
	for i in range(n):
		if a[i] == 0 and a[(i + 1) % n] == 0:
			print("Yes")
			return
			
	print("No")

if __name__ == '__main__':
	main()