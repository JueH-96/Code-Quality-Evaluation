import heapq

def main():
	import sys
	data = sys.stdin.read().splitlines()
	q = int(data[0].strip())
	total_add = 0
	heap = []
	output_lines = []
	index = 1
	for _ in range(q):
		parts = data[index].split()
		index += 1
		if parts[0] == '1':
			heapq.heappush(heap, total_add)
		elif parts[0] == '2':
			T = int(parts[1])
			total_add += T
		elif parts[0] == '3':
			H = int(parts[1])
			threshold = total_add - H
			count = 0
			while heap and heap[0] <= threshold:
				heapq.heappop(heap)
				count += 1
			output_lines.append(str(count))
	
	print("
".join(output_lines))

if __name__ == "__main__":
	main()