import sys
from collections import deque

def max_pyramid_size(N, A):
    left = deque()
    right = deque()

    for i in range(N):
        if not left or A[i] >= left[-1]:
            left.append(A[i])
        else:
            while left and A[i] < left[-1]:
                right.appendleft(left.pop())
            left.append(A[i])

    max_size = 0
    while left and right:
        if left[0] < right[0]:
            left.popleft()
        elif left[0] > right[0]:
            right.popleft()
        else:
            max_size += 2
            left.popleft()
            right.popleft()

    if left:
        max_size += 1
    if right:
        max_size += 1

    return max_size

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    print(max_pyramid_size(N, A))

if __name__ == "__main__":
    main()