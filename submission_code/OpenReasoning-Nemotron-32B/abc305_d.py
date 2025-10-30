import bisect
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	queries = []
	index = 1 + n + 1
	for i in range(q):
		l = int(data[index])
		r = int(data[index+1])
		index += 2
		queries.append((l, r))
	
	k = (n - 1) // 2
	S = []
	E = []
	for i in range(1, n, 2):
		if i + 1 < n:
			S.append(A[i])
			E.append(A[i+1])
	
	prefix = [0] * (k + 1)
	for i in range(1, k + 1):
		prefix[i] = prefix[i-1] + (E[i-1] - S[i-1])
	
	out_lines = []
	for (l, r) in queries:
		i_start = bisect.bisect_right(E, l)
		i_end = bisect.bisect_left(S, r) - 1
		
		if i_start > i_end:
			out_lines.append("0")
		else:
			total_block = prefix[i_end + 1] - prefix[i_start]
			subtract_left = max(0, l - S[i_start])
			subtract_right = max(0, E[i_end] - r)
			ans = total_block - subtract_left - subtract_right
			out_lines.append(str(ans))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()