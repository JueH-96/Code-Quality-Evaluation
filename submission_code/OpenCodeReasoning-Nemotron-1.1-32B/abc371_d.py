import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	X = list(map(int, data[1:1+n]))
	P = list(map(int, data[1+n:1+2*n]))
	q = int(data[1+2*n])
	queries = []
	index = 1 + 2*n + 1
	for i in range(q):
		L = int(data[index])
		R = int(data[index+1])
		index += 2
		queries.append((L, R))
	
	prefix = [0] * (n+1)
	for i in range(1, n+1):
		prefix[i] = prefix[i-1] + P[i-1]
	
	results = []
	for L, R in queries:
		left_idx = bisect.bisect_left(X, L)
		right_idx = bisect.bisect_right(X, R) - 1
		if left_idx > right_idx:
			results.append("0")
		else:
			total = prefix[right_idx+1] - prefix[left_idx]
			results.append(str(total))
	
	print("
".join(results))

if __name__ == "__main__":
	main()