import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	
	left_ones = [0] * n
	if n > 0:
		if s[0] == '1':
			left_ones[0] = 1
		else:
			left_ones[0] = 0
			
		for i in range(1, n):
			if s[i] == '1':
				left_ones[i] = left_ones[i-1] + 1
			else:
				left_ones[i] = 0
				
	right_twos = [0] * n
	if n > 0:
		if s[-1] == '2':
			right_twos[-1] = 1
		else:
			right_twos[-1] = 0
			
		for i in range(n-2, -1, -1):
			if s[i] == '2':
				right_twos[i] = right_twos[i+1] + 1
			else:
				right_twos[i] = 0
				
	ans = 0
	for i in range(n):
		if s[i] == '/':
			left_count = left_ones[i-1] if i-1 >= 0 else 0
			right_count = right_twos[i+1] if i+1 < n else 0
			candidate = 2 * min(left_count, right_count) + 1
			if candidate > ans:
				ans = candidate
				
	print(ans)

if __name__ == "__main__":
	main()