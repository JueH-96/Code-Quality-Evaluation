import sys
import heapq

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index])
		m = int(data[index + 1])
		index += 2
		boxes = []
		for _ in range(n):
			v = int(data[index])
			p = int(data[index + 1])
			index += 2
			boxes.append((p, v))
		
		boxes.sort()
		capacities = []
		for p, v in boxes:
			if p <= v:
				capacities.append(v)
		
		capacities.sort()
		total_balls = 0
		heap = []
		j = 0
		for i in range(1, m + 1):
			while j < len(capacities) and capacities[j] < i:
				heapq.heappush(heap, -capacities[j])
				j += 1
			if heap:
				total_balls += -heapq.heappop(heap)
			else:
				break
		
		total_cost = 0
		selected_boxes = []
		for p, v in boxes:
			if p <= v and v >= 1:
				selected_boxes.append(p)
				if len(selected_boxes) == len(heap) + j:
					break
		total_cost = sum(selected_boxes[:len(heap) + j])
		
		results.append(str(total_balls - total_cost))
	
	print("
".join(results))

if __name__ == "__main__":
	main()