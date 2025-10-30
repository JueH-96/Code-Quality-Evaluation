import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	P = list(map(int, data[2:2+n]))
	
	pos = [0] * (n + 1)
	for i in range(n):
		pos[P[i]] = i + 1
		
	A = [0] * n
	for i in range(1, n + 1):
		A[i - 1] = pos[i]
		
	if k == 1:
		print(0)
		return
		
	min_deque = deque()
	max_deque = deque()
	ans = 10**18
	
	for i in range(n):
		while min_deque and A[min_deque[-1]] > A[i]:
			min_deque.pop()
		min_deque.append(i)
		
		while max_deque and A[max_deque[-1]] < A[i]:
			max_deque.pop()
		max_deque.append(i)
		
		if min_deque and min_deque[0] <= i - k:
			min_deque.popleft()
		if max_deque and max_deque[0] <= i - k:
			max_deque.popleft()
			
		if i >= k - 1:
			min_val = A[min_deque[0]]
			max_val = A[max_deque[0]]
			candidate = max_val - min_val
			if candidate < ans:
				ans = candidate
				
	print(ans)

if __name__ == "__main__":
	main()