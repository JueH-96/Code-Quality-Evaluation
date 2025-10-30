import sys
import heapq

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	A = [int(next(it)) for _ in range(n)]
	size = n + 1
	freq = [0] * size

	for a in A:
		if a < size:
			freq[a] += 1

	heap = []
	for i in range(size):
		if freq[i] == 0:
			heapq.heappush(heap, i)
			
	out_lines = []
	for _ in range(q):
		i_val = int(next(it))
		x_val = int(next(it))
		pos = i_val - 1
		old_val = A[pos]
		A[pos] = x_val

		if old_val < size:
			freq[old_val] -= 1
			if freq[old_val] == 0:
				heapq.heappush(heap, old_val)
				
		if x_val < size:
			freq[x_val] += 1

		while heap and freq[heap[0]] != 0:
			heapq.heappop(heap)
			
		mex = heap[0]
		out_lines.append(str(mex))
		
	sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
	main()