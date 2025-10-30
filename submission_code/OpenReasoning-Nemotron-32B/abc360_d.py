import sys
import bisect

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	T = int(data[1])
	s = data[2]
	xs = list(map(int, data[3:3+n]))
	
	A = []   # for ants moving right (S_i = '1')
	B = []   # for ants moving left (S_i = '0')
	
	for i in range(n):
		if s[i] == '1':
			A.append(xs[i])
		else:
			B.append(xs[i])
			
	A.sort()
	B.sort()
	
	total = 0
	for a in A:
		low_bound = a
		high_bound = a + 2 * T
		left_index = bisect.bisect_left(B, low_bound)
		right_index = bisect.bisect_right(B, high_bound)
		total += right_index - left_index
		
	print(total)

if __name__ == "__main__":
	main()