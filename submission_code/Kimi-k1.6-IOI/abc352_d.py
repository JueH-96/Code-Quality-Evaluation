import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    pos = [0] * (N + 1)
    for i in range(N):
        pos[P[i]] = i + 1
    A = [pos[i] for i in range(1, N+1)]
    
    min_diff = float('inf')
    min_deque = deque()
    max_deque = deque()
    
    for i in range(len(A)):
        # Maintain min_deque in increasing order
        while min_deque and A[i] <= A[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        
        # Maintain max_deque in decreasing order
        while max_deque and A[i] >= A[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        
        # Remove elements out of the current window's left bound
        while min_deque[0] < i - K + 1:
            min_deque.popleft()
        while max_deque[0] < i - K + 1:
            max_deque.popleft()
        
        # Calculate the current window's min and max difference
        if i >= K - 1:
            current_min = A[min_deque[0]]
            current_max = A[max_deque[0]]
            diff = current_max - current_min
            if diff < min_diff:
                min_diff = diff
    print(min_diff)

if __name__ == '__main__':
    main()