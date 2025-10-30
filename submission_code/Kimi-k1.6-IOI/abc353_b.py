from collections import deque
import sys

def main():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    q = deque(a)
    count = 0
    seats = k
    while True:
        if not q:
            count += 1
            break
        if seats >= q[0]:
            seats -= q.popleft()
        else:
            count += 1
            seats = k
    print(count)

if __name__ == "__main__":
    main()