import sys

def solve():
    H, W, N = map(int, input().split())
    bars = []
    for _ in range(N):
        R, C, L = map(int, input().split())
        bars.append((R, C, L))

    for _ in range(100):
        for i in range(N):
            R, C, L = bars[i]
            if R < H and all(bars[j][1] != C + k for j in range(N) for k in range(L)):
                bars[i] = (R + 1, C, L)

    return [bar[0] for bar in bars]

print('
'.join(map(str, solve())))