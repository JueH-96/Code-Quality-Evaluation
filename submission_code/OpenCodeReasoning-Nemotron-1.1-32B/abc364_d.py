import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	q = int(data[1])
	A = list(map(int, data[2:2+n]))
	A.sort()
	queries = []
	index = 2 + n
	for i in range(q):
		b = int(data[index])
		k = int(data[index+1])
		index += 2
		queries.append((b, k))
	
	out_lines = []
	for b, k in queries:
		lo = 0
		hi = 200000000
		while lo < hi:
			mid = (lo + hi) // 2
			left_bound = b - mid
			right_bound = b + mid
			l_idx = bisect.bisect_left(A, left_bound)
			r_idx = bisect.bisect_right(A, right_bound)
			count = r_idx - l_idx
			if count >= k:
				hi = mid
			else:
				lo = mid + 1
		out_lines.append(str(lo))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()