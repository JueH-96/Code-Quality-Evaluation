def main():
	n = int(input().strip())
	s = input().strip()
	
	if n % 2 == 0:
		print("No")
		return
		
	mid = n // 2
	if s[mid] != '/':
		print("No")
		return
		
	first_half = s[:mid]
	second_half = s[mid+1:]
	
	if first_half == '1' * mid and second_half == '2' * mid:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()