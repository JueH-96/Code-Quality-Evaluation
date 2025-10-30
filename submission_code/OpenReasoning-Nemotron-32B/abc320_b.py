def main():
	s = input().strip()
	n = len(s)
	max_len = 1  # At least one character is always a palindrome
	
	for i in range(n):
		# Check for odd-length palindromes centered at i
		left, right = i, i
		while left >= 0 and right < n and s[left] == s[right]:
			left -= 1
			right += 1
		len_odd = right - left - 1
		
		# Check for even-length palindromes centered between i and i+1
		left, right = i, i + 1
		while left >= 0 and right < n and s[left] == s[right]:
			left -= 1
			right += 1
		len_even = right - left - 1
		
		# Update the maximum length found
		if len_odd > max_len:
			max_len = len_odd
		if len_even > max_len:
			max_len = len_even
	
	print(max_len)

if __name__ == '__main__':
	main()