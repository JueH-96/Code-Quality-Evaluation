import sys
from collections import deque

def main():
    n, k = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    
    pos = [0] * (n + 1)  # pos[x] is the 1-based index of x in p
    for idx in range(n):
        val = p[idx]
        pos[val] = idx + 1  # Convert to 1-based index
    
    if k == 1:
        print(0)
        return
    
    min_deque = deque()
    max_deque = deque()
    min_answer = float('inf')
    
    for i in range(1, n + 1):
        # Maintain min_deque (increasing order)
        while min_deque and pos[min_deque[-1]] >= pos[i]:
            min_deque.pop()
        min_deque.append(i)
        
        # Maintain max_deque (decreasing order)
        while max_deque and pos[max_deque[-1]] <= pos[i]:
            max_deque.pop()
        max_deque.append(i)
        
        # Check if the window of size k is formed
        if i >= k:
            a = i - k + 1
            # Remove elements out of the current window from the front
            while min_deque[0] < a:
                min_deque.popleft()
            while max_deque[0] < a:
                max_deque.popleft()
            
            current_min = pos[min_deque[0]]
            current_max = pos[max_deque[0]]
            current_diff = current_max - current_min
            if current_diff < min_answer:
                min_answer = current_diff
    
    print(min_answer)

if __name__ == "__main__":
    main()