import sys
import bisect

def main():
	input = sys.stdin.readline
	n, q = map(int, input().split())
	A = list(map(int, input().split()))
	A.sort()
	out_lines = []
	for _ in range(q):
		b, k = map(int, input().split())
		low = 0
		d1 = abs(A[0] - b)
		d2 = abs(A[-1] - b)
		high = max(d1, d2)
		
		while low < high:
			mid = (low + high) // 2
			left_bound = b - mid
			right_bound = b + mid
			l_index = bisect.bisect_left(A, left_bound)
			r_index = bisect.bisect_right(A, right_bound)
			count = r_index - l_index
			
			if count >= k:
				high = mid
			else:
				low = mid + 1
				
		out_lines.append(str(low))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()