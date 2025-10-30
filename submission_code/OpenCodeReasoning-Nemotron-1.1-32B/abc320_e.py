import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	events = []
	index = 2
	for i in range(m):
		t = int(data[index])
		w = int(data[index+1])
		s = int(data[index+2])
		index += 3
		events.append((t, w, s))
	
	ans = [0] * n
	returns = []
	row_heap = []
	in_row = [True] * n
	
	for i in range(n):
		heapq.heappush(row_heap, i)
	
	for t, w, s in events:
		while returns and returns[0][0] <= t:
			return_time, p = heapq.heappop(returns)
			if not in_row[p]:
				in_row[p] = True
				heapq.heappush(row_heap, p)
				
		if row_heap:
			p = heapq.heappop(row_heap)
			in_row[p] = False
			ans[p] += w
			heapq.heappush(returns, (t + s, p))
			
	for a in ans:
		print(a)

if __name__ == "__main__":
	main()