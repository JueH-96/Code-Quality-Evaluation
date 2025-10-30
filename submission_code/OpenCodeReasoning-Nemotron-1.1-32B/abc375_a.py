def main():
	N = int(input().strip())
	S = input().strip()
	
	if N < 3:
		print(0)
		return
		
	count = 0
	for i in range(N - 2):
		if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
			count += 1
			
	print(count)

if __name__ == "__main__":
	main()