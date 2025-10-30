import sys

def max_height():
    N = int(input())
    heights = []
    for _ in range(N):
        a, b = map(int, input().split())
        heights.append((a, b))

    heights.sort(key=lambda x: x[1])
    max_height = 0
    current_height = 0
    for a, b in heights:
        current_height += a
        max_height = max(max_height, current_height + b)

    print(max_height)

max_height()