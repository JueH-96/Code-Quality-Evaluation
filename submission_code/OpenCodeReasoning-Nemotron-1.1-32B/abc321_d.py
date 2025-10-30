import sys
import bisect

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	p = int(data[2])
	A = list(map(int, data[3:3+n]))
	B = list(map(int, data[3+n:3+n+m]))
	
	B.sort()
	prefix = [0] * (m + 1)
	for i in range(1, m + 1):
		prefix[i] = prefix[i - 1] + B[i - 1]
	
	total_pairs = n * m
	total_count = 0
	term1 = 0
	
	for a in A:
		x = p - a
		if x < 0:
			continue
		idx = bisect.bisect_right(B, x)
		count_i = idx
		sum_i = prefix[idx]
		term1 += a * count_i + sum_i
		total_count += count_i
	
	term2 = p * (total_pairs - total_count)
	ans = term1 + term2
	print(ans)

if __name__ == "__main__":
	main()