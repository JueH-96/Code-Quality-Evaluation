import sys

def main():
	data = sys.stdin.read().splitlines()
	N = int(data[0].strip())
	S = data[1].strip()
	
	freq_S = [0] * 10
	for char in S:
		d = ord(char) - ord('0')
		freq_S[d] += 1
		
	max_val = 10 ** N
	count = 0
	x = 0
	while x * x < max_val:
		num_str = str(x*x)
		num_str_padded = num_str.zfill(N)
		freq_candidate = [0] * 10
		for c in num_str_padded:
			digit = ord(c) - ord('0')
			freq_candidate[digit] += 1
			
		if freq_candidate == freq_S:
			count += 1
			
		x += 1
		
	print(count)

if __name__ == '__main__':
	main()