import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    H, W, Q = map(int, input().split())
    N = H * W

    # For each cell (flattened id = r*W + c), store neighbor pointers or -1 if none.
    left = [-1] * N
    right = [-1] * N
    up = [-1] * N
    down = [-1] * N

    # Initialize pointers
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if c > 0:
                left[i] = i - 1
            if c < W - 1:
                right[i] = i + 1
            if r > 0:
                up[i] = i - W
            if r < H - 1:
                down[i] = i + W

    destroyed = [False] * N
    remaining = N

    def remove_cell(i):
        nonlocal remaining
        if destroyed[i]:
            return
        l = left[i]
        r_ = right[i]
        u = up[i]
        d = down[i]
        if l != -1:
            right[l] = r_
        if r_ != -1:
            left[r_] = l
        if u != -1:
            down[u] = d
        if d != -1:
            up[d] = u
        destroyed[i] = True
        remaining -= 1

    for _ in range(Q):
        r, c = map(int, input().split())
        r -= 1
        c -= 1
        idx = r * W + c
        if not destroyed[idx]:
            # destroy this wall
            remove_cell(idx)
        else:
            # already empty: destroy the first in each direction
            targets = []
            if up[idx] != -1:
                targets.append(up[idx])
            if down[idx] != -1:
                targets.append(down[idx])
            if left[idx] != -1:
                targets.append(left[idx])
            if right[idx] != -1:
                targets.append(right[idx])
            for t in targets:
                remove_cell(t)

    print(remaining)

if __name__ == "__main__":
    main()