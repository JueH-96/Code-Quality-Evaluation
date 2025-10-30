import bisect

mod = 10**8

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	total_sum = sum(A)
	total_pairs_sum = (n - 1) * total_sum
	
	A.sort()
	total_count = 0
	for i in range(n):
		target = mod - A[i]
		j0 = bisect.bisect_left(A, target, i + 1)
		total_count += (n - j0)
	
	answer = total_pairs_sum - mod * total_count
	print(answer)

if __name__ == '__main__':
	main()