import sys
import heapq

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return

	n = int(data[0])
	m = int(data[1])
	D = int(data[2])
	A = list(map(int, data[3:3+n]))
	B = list(map(int, data[3+n:3+n+m]))
	
	A.sort()
	B.sort()
	
	l = 0
	r = 0
	heap = []
	removed = {}
	best = -1

	for a in A:
		while l < m and B[l] < a - D:
			val = B[l]
			removed[val] = removed.get(val, 0) + 1
			l += 1
			
		if r < l:
			r = l
			
		while r < m and B[r] <= a + D:
			heapq.heappush(heap, -B[r])
			r += 1
			
		while heap:
			top_val = -heap[0]
			if top_val in removed and removed[top_val] > 0:
				heapq.heappop(heap)
				removed[top_val] -= 1
				if removed[top_val] == 0:
					del removed[top_val]
			else:
				break
				
		if heap:
			total = a + (-heap[0])
			if total > best:
				best = total
				
	print(best)

if __name__ == "__main__":
	main()