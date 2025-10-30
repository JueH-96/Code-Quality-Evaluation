import sys
import bisect

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
	
	n_intervals = (n - 1) // 2
	starts = []
	ends = []
	for i in range(1, n, 2):
		starts.append(A[i])
		ends.append(A[i+1])
	
	prefix_sum_full = [0] * (n_intervals + 1)
	for i in range(1, n_intervals + 1):
		prefix_sum_full[i] = prefix_sum_full[i-1] + (ends[i-1] - starts[i-1])
	
	def f(x):
		j = bisect.bisect_right(ends, x)
		full_sum = prefix_sum_full[j]
		i_index = bisect.bisect_right(starts, x) - 1
		if i_index >= 0 and ends[i_index] > x:
			full_sum += x - starts[i_index]
		return full_sum
	
	results = []
	for l, r in queries:
		res = f(r) - f(l)
		results.append(str(res))
	
	sys.stdout.write("
".join(results))

if __name__ == "__main__":
	main()