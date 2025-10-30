import sys
import bisect

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	X = list(map(int, data[1:1+n]))
	P = list(map(int, data[1+n:1+2*n]))
	q = int(data[1+2*n])
	queries = []
	index = 1+2*n+1
	for i in range(q):
		L = int(data[index])
		R = int(data[index+1])
		index += 2
		queries.append((L, R))
	
	prefix = [0] * (n+1)
	for i in range(1, n+1):
		prefix[i] = prefix[i-1] + P[i-1]
	
	out_lines = []
	for L, R in queries:
		left_index = bisect.bisect_left(X, L)
		right_index = bisect.bisect_right(X, R) - 1
		if left_index > right_index:
			out_lines.append("0")
		else:
			total = prefix[right_index+1] - prefix[left_index]
			out_lines.append(str(total))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()