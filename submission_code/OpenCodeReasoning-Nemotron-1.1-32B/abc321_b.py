def main():
	import sys
	input = sys.stdin.read().split()
	n = int(input[0])
	x = int(input[1])
	A = list(map(int, input[2:2+n-1]))
	
	min_A = min(A)
	max_A = max(A)
	sum_A = sum(A)
	
	candidate = 101
	
	if sum_A - max_A >= x:
		candidate = 0
		
	if sum_A - min_A >= x:
		candidate = min(candidate, max_A)
		
	if min_A < max_A:
		k = x - (sum_A - min_A - max_A)
		s0 = max(min_A + 1, k)
		if s0 < max_A:
			candidate = min(candidate, s0)
			
	print(-1 if candidate == 101 else candidate)

if __name__ == "__main__":
	main()