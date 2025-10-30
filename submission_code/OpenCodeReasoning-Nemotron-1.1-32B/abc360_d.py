import sys
import bisect

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	n, T = map(int, data[0].split())
	s = data[1].strip()
	xs = list(map(int, data[2].split()))
	
	A = []
	B = []
	
	for i in range(n):
		if s[i] == '1':
			A.append(xs[i])
		else:
			B.append(xs[i])
			
	A.sort()
	B.sort()
	
	if not B:
		print(0)
		return
		
	count = 0
	for a in A:
		low_bound = a + 1
		if low_bound > B[-1]:
			break
		high_bound = a + 2 * T
		if high_bound < B[0]:
			continue
			
		left_index = bisect.bisect_left(B, low_bound)
		right_index = bisect.bisect_right(B, high_bound) - 1
		
		if left_index <= right_index:
			count += (right_index - left_index + 1)
			
	print(count)

if __name__ == '__main__':
	main()