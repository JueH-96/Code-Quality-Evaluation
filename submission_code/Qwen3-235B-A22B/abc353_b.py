import sys
from collections import deque

def main():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    q = deque(a)
    count = 0
    current_seats = k

    while True:
        if not q:
            count += 1
            break
        front = q[0]
        if current_seats < front:
            count += 1
            current_seats = k
        else:
            current_seats -= front
            q.popleft()
    
    print(count)

if __name__ == "__main__":
    main()