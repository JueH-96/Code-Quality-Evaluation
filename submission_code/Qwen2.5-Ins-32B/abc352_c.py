import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    idx = 1
    giants = []
    for _ in range(n):
        a = int(data[idx])
        b = int(data[idx + 1])
        giants.append((a, b))
        idx += 2
    giants.sort(key=lambda x: (x[1] - x[0], x[0]), reverse=True)
    height = 0
    for a, b in giants:
        height += a
        height += b - a
    print(height)

solve()