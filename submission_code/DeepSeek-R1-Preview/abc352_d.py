import sys
from collections import deque

def main():
    n, k = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    
    pos = [0] * (n + 1)  # pos[x] gives the position of x in the permutation (1-based)
    for i in range(n):
        x = p[i]
        pos[x] = i + 1  # positions are 1-based
    
    max_deque = deque()
    min_deque = deque()
    min_diff = float('inf')
    
    for i in range(1, n + 1):
        # Maintain max_deque
        while max_deque and pos[i] >= pos[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        
        # Maintain min_deque
        while min_deque and pos[i] <= pos[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        
        # Remove elements outside the window [i - k + 1, i]
        while max_deque[0] < (i - k + 1):
            max_deque.popleft()
        while min_deque[0] < (i - k + 1):
            min_deque.popleft()
        
        # Calculate the difference if window size is k
        if i >= k:
            current_max = pos[max_deque[0]]
            current_min = pos[min_deque[0]]
            diff = current_max - current_min
            if diff < min_diff:
                min_diff = diff
    
    print(min_diff)

if __name__ == "__main__":
    main()