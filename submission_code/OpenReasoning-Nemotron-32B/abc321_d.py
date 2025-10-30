import sys
import bisect

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	m = int(data[1])
	p = int(data[2])
	A = list(map(int, data[3:3+n]))
	B = list(map(int, data[3+n:3+n+m]))
	
	total_A = sum(A)
	total_B = sum(B)
	first_term = m * total_A + n * total_B
	
	B.sort()
	prefix = [0] * (m + 1)
	for i in range(1, m + 1):
		prefix[i] = prefix[i-1] + B[i-1]
	
	T = 0
	for a in A:
		x = p - a
		k = bisect.bisect_right(B, x)
		count = m - k
		sum_suffix = prefix[m] - prefix[k]
		T += (a - p) * count + sum_suffix
	
	ans = first_term - T
	print(ans)

if __name__ == "__main__":
	main()