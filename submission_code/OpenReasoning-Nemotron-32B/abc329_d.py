import heapq
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	votes = list(map(int, data[2:2+m]))
	
	count = [0] * (n + 1)
	heap = []
	out_lines = []
	
	for a in votes:
		count[a] += 1
		heapq.heappush(heap, (-count[a], a))
		
		while heap:
			neg_count, cand = heap[0]
			if -neg_count != count[cand]:
				heapq.heappop(heap)
			else:
				break
				
		winner = heap[0][1]
		out_lines.append(str(winner))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()