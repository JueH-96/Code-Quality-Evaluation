from collections import deque
import sys

def main():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    groups = deque(a)
    current_empty = k
    count = 0
    while True:
        if not groups:
            count += 1
            break
        if current_empty >= groups[0]:
            current_empty -= groups.popleft()
        else:
            count += 1
            current_empty = k
    print(count)

if __name__ == "__main__":
    main()