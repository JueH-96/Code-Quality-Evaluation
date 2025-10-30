def main():
	n = int(input().strip())
	s = input().strip()
	
	if n == 0:
		print(0)
		return
		
	if s[0] == '0':
		ones = 0
		zeros = 1
	else:
		ones = 1
		zeros = 0
		
	total = ones
	
	for i in range(1, n):
		if s[i] == '0':
			ones_curr = i
			zeros_curr = 1
		else:
			ones_curr = zeros + 1
			zeros_curr = ones
			
		total += ones_curr
		ones, zeros = ones_curr, zeros_curr
		
	print(total)

if __name__ == "__main__":
	main()