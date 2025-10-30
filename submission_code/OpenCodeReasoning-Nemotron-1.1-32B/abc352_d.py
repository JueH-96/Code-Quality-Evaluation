import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	p = list(map(int, data[2:2+n]))
	
	pos = [0] * (n + 1)
	for i in range(n):
		pos[p[i]] = i + 1
	
	min_heap = []
	max_heap = []
	min_removed = [0] * (n + 2)
	max_removed = [0] * (n + 2)
	
	for num in range(1, k + 1):
		idx = pos[num]
		heapq.heappush(min_heap, idx)
		heapq.heappush(max_heap, -idx)
	
	ans = float('inf')
	
	for a in range(1, n - k + 2):
		while min_heap:
			top = min_heap[0]
			if min_removed[top] > 0:
				heapq.heappop(min_heap)
				min_removed[top] -= 1
			else:
				break
		
		while max_heap:
			top_neg = max_heap[0]
			top_val = -top_neg
			if max_removed[top_val] > 0:
				heapq.heappop(max_heap)
				max_removed[top_val] -= 1
			else:
				break
		
		min_index = min_heap[0]
		max_index = -max_heap[0]
		candidate = max_index - min_index
		if candidate < ans:
			ans = candidate
		
		if a == n - k + 1:
			break
		
		remove_idx = pos[a]
		min_removed[remove_idx] += 1
		max_removed[remove_idx] += 1
		
		add_idx = pos[a + k]
		heapq.heappush(min_heap, add_idx)
		heapq.heappush(max_heap, -add_idx)
	
	print(ans)

if __name__ == "__main__":
	main()