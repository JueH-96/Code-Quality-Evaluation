import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	M = int(next(it))
	
	events = []
	for _ in range(M):
		T = int(next(it))
		W = int(next(it))
		S = int(next(it))
		heapq.heappush(events, (T, 1, W, S))
	
	row_heap = list(range(1, N+1))
	heapq.heapify(row_heap)
	
	ans = [0] * (N+1)
	
	while events:
		time, etype, *rest = heapq.heappop(events)
		if etype == 0:
			person = rest[0]
			heapq.heappush(row_heap, person)
		else:
			W_val = rest[0]
			S_val = rest[1]
			if row_heap:
				person = heapq.heappop(row_heap)
				ans[person] += W_val
				return_time = time + S_val
				heapq.heappush(events, (return_time, 0, person))
	
	for i in range(1, N+1):
		print(ans[i])

if __name__ == "__main__":
	main()