def main():
	data = input().split()
	n = int(data[0])
	c1 = data[1]
	c2 = data[2]
	S = input().strip()
	
	result = ''.join(c1 if char == c1 else c2 for char in S)
	print(result)

if __name__ == "__main__":
	main()