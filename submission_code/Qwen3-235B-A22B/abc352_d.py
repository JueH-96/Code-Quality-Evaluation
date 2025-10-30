import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    pos = [0] * (N + 2)
    for i in range(N):
        x = P[i]
        pos[x] = i + 1  # Positions are 1-based
    A = [pos[x] for x in range(1, N+1)]
    
    if K == 1:
        print(0)
        return
    
    max_deque = deque()
    min_deque = deque()
    result = float('inf')
    
    for i in range(N):
        # Update max deque
        while max_deque and A[i] >= A[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        
        # Update min deque
        while min_deque and A[i] <= A[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        
        # Calculate window_left
        window_left = i - K + 1
        
        # Remove elements out of window
        while max_deque and max_deque[0] < window_left:
            max_deque.popleft()
        while min_deque and min_deque[0] < window_left:
            min_deque.popleft()
        
        # Check if window is valid
        if i >= K - 1:
            current_max = A[max_deque[0]]
            current_min = A[min_deque[0]]
            diff = current_max - current_min
            if diff < result:
                result = diff
    
    print(result)

if __name__ == "__main__":
    main()