import sys
from collections import deque

def main():
    n, k = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    pos = [0] * (n + 1)
    for i in range(n):
        pos[p[i]] = i + 1

    min_deque = deque()
    max_deque = deque()
    min_diff = float('inf')

    for i in range(1, n + 1):
        # Update min deque
        while min_deque and pos[i] <= pos[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        # Remove elements out of the window [i-k+1, i]
        while min_deque[0] < i - k + 1:
            min_deque.popleft()

        # Update max deque
        while max_deque and pos[i] >= pos[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        # Remove elements out of the window [i-k+1, i]
        while max_deque[0] < i - k + 1:
            max_deque.popleft()

        # Check if the current window is valid (i >= k)
        if i >= k:
            current_min = pos[min_deque[0]]
            current_max = pos[max_deque[0]]
            min_diff = min(min_diff, current_max - current_min)

    print(min_diff)

if __name__ == "__main__":
    main()