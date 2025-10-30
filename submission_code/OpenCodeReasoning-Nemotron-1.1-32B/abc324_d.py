import math

def main():
	n = int(input().strip())
	s = input().strip()
	
	if n == 0:
		print(0)
		return
		
	freq_s = [0] * 10
	for char in s:
		d = int(char)
		freq_s[d] += 1
		
	max_val = 10**n - 1
	max_k = math.isqrt(max_val)
	
	count = 0
	for k in range(0, max_k + 1):
		square = k * k
		freq = [0] * 10
		temp = square
		valid = True
		for i in range(n):
			digit = temp % 10
			freq[digit] += 1
			if freq[digit] > freq_s[digit]:
				valid = False
				break
			temp //= 10
		if not valid:
			continue
		if freq == freq_s:
			count += 1
			
	print(count)

if __name__ == '__main__':
	main()