import heapq
import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	out_lines = []
	for _ in range(t):
		n = int(data[index]); k = int(data[index+1]); index += 2
		A = list(map(int, data[index:index+n])); index += n
		B = list(map(int, data[index:index+n])); index += n
		
		if n == 0:
			out_lines.append("0")
			continue
			
		arr = list(zip(A, B))
		arr.sort(key=lambda x: x[0])
		
		heap = []
		total_sum = 0
		ans = 10**18
		
		i = 0
		while i < n:
			j = i
			min_current_group = arr[i][1]
			while j < n and arr[j][0] == arr[i][0]:
				b_val = arr[j][1]
				if len(heap) < k:
					heapq.heappush(heap, -b_val)
					total_sum += b_val
				else:
					if b_val < -heap[0]:
						removed = -heapq.heappop(heap)
						total_sum -= removed
						heapq.heappush(heap, -b_val)
						total_sum += b_val
				min_current_group = min(min_current_group, b_val)
				j += 1
				
			if len(heap) < k:
				i = j
				continue
				
			if min_current_group <= -heap[0]:
				candidate_sum = total_sum
			else:
				candidate_sum = total_sum - (-heap[0]) + min_current_group
				
			candidate_value = arr[i][0] * candidate_sum
			if candidate_value < ans:
				ans = candidate_value
				
			i = j
			
		out_lines.append(str(ans))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()