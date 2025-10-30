import heapq
import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	A = [int(next(it)) for _ in range(n)]
	
	freq = [0] * (n + 1)
	
	for a in A:
		if 0 <= a <= n:
			freq[a] += 1
			
	heap = []
	for i in range(0, n + 1):
		if freq[i] == 0:
			heapq.heappush(heap, i)
			
	out_lines = []
	for _ in range(q):
		i_val = int(next(it))
		x_val = int(next(it))
		pos = i_val - 1
		
		old_val = A[pos]
		new_val = x_val
		A[pos] = new_val
		
		if 0 <= old_val <= n:
			freq[old_val] -= 1
			if freq[old_val] == 0:
				heapq.heappush(heap, old_val)
				
		if 0 <= new_val <= n:
			freq[new_val] += 1
			
		while heap:
			top = heap[0]
			if freq[top] > 0:
				heapq.heappop(heap)
			else:
				break
				
		mex = heap[0] if heap else n + 1
		out_lines.append(str(mex))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()