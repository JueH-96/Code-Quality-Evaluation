import heapq
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	left_initial = [i-1 for i in range(n)]
	right_initial = [i+1 for i in range(n)]
	if n > 0:
		right_initial[-1] = -1
	
	ans = [0] * n
	
	for start in range(n):
		current = A[start]
		left_bound = start - 1
		right_bound = start + 1
		heap = []
		absorbed = set()
		absorbed.add(start)
		
		if left_bound >= 0:
			heapq.heappush(heap, (A[left_bound], left_bound, 'L'))
		if right_bound < n:
			heapq.heappush(heap, (A[right_bound], right_bound, 'R'))
			
		while heap:
			val, index, direction = heapq.heappop(heap)
			if index in absorbed:
				continue
			if val >= current:
				break
				
			current += val
			absorbed.add(index)
			
			if direction == 'L':
				new_left = left_initial[index]
				left_bound = new_left
				if new_left >= 0 and new_left not in absorbed:
					heapq.heappush(heap, (A[new_left], new_left, 'L'))
			else:
				new_right = right_initial[index]
				right_bound = new_right
				if new_right < n and new_right not in absorbed:
					heapq.heappush(heap, (A[new_right], new_right, 'R'))
					
		ans[start] = current
		
	print(" ".join(map(str, ans)))

if __name__ == "__main__":
	main()