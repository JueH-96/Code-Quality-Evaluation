import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    queue = deque(A)
    current_seats = K
    count = 0

    while True:
        if not queue:
            count += 1
            break
        front = queue[0]
        if current_seats < front:
            count += 1
            current_seats = K
        else:
            current_seats -= front
            queue.popleft()
    print(count)

if __name__ == "__main__":
    main()