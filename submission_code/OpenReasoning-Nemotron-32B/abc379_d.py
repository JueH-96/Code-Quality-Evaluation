import heapq
import sys

def main():
	input = sys.stdin.readline
	Q = int(input())
	heap = []
	global_time = 0
	output_lines = []
	
	for _ in range(Q):
		data = input().split()
		if data[0] == '1':
			heapq.heappush(heap, global_time)
		elif data[0] == '2':
			T = int(data[1])
			global_time += T
		elif data[0] == '3':
			H = int(data[1])
			threshold = global_time - H
			cnt = 0
			while heap and heap[0] <= threshold:
				heapq.heappop(heap)
				cnt += 1
			output_lines.append(str(cnt))
	
	print("
".join(output_lines))

if __name__ == "__main__":
	main()