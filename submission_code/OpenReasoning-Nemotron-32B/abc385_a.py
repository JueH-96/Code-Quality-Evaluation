def main():
	data = input().split()
	a = int(data[0])
	b = int(data[1])
	c = int(data[2])
	
	if a == b == c:
		print("Yes")
	elif a == b + c or b == a + c or c == a + b:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()