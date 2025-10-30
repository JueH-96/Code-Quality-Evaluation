import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    pos = [0] * (N + 2)  # 1-based to N
    for i in range(N):
        pos[P[i]] = i + 1  # positions are 1-based
    arr = pos[1:N+1]
    
    if K == 1:
        print(0)
        return
    
    # Sliding window min
    min_deque = deque()
    min_values = []
    for i in range(len(arr)):
        while min_deque and arr[i] <= arr[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        while min_deque[0] <= i - K:
            min_deque.popleft()
        if i >= K - 1:
            min_values.append(arr[min_deque[0]])
    
    # Sliding window max
    max_deque = deque()
    max_values = []
    for i in range(len(arr)):
        while max_deque and arr[i] >= arr[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        while max_deque[0] <= i - K:
            max_deque.popleft()
        if i >= K - 1:
            max_values.append(arr[max_deque[0]])
    
    # Find minimal difference
    min_diff = float('inf')
    for hm in zip(min_values, max_values):
        diff = hm[1] - hm[0]
        if diff < min_diff:
            min_diff = diff
    print(min_diff)

if __name__ == "__main__":
    main()