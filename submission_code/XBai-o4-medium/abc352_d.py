import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Build the pos array
    pos = [0] * (N + 1)  # pos[0] is unused
    for i in range(N):
        x = P[i]
        pos[x] = i + 1  # 1-based index

    arr = pos[1:]  # Now arr is [pos[1], pos[2], ..., pos[N]]

    if K == 1:
        print(0)
        return

    # Sliding window min and max
    min_deque = deque()
    max_deque = deque()
    min_list = []
    max_list = []

    for i in range(len(arr)):
        # Process min_deque
        while min_deque and arr[i] <= arr[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(i)
        # Remove elements out of the window
        while min_deque[0] <= i - K:
            min_deque.popleft()
        # Process max_deque
        while max_deque and arr[i] >= arr[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(i)
        while max_deque[0] <= i - K:
            max_deque.popleft()
        # Check if current index is the end of a window
        if i >= K - 1:
            min_val = arr[min_deque[0]]
            max_val = arr[max_deque[0]]
            min_list.append(min_val)
            max_list.append(max_val)

    # Find the minimum difference
    min_diff = float('inf')
    for i in range(len(min_list)):
        current_diff = max_list[i] - min_list[i]
        if current_diff < min_diff:
            min_diff = current_diff

    print(min_diff)

if __name__ == '__main__':
    main()