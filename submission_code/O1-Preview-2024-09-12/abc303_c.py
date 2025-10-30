# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    items = set()
    for _ in range(M):
        xi, yi = map(int, sys.stdin.readline().split())
        items.add((xi, yi))
    x, y = 0, 0
    for c in S:
        if c == 'R':
            x += 1
        elif c == 'L':
            x -=1
        elif c == 'U':
            y +=1
        elif c == 'D':
            y -=1
        H -=1
        if H < 0:
            print("No")
            return
        if H < K and (x, y) in items:
            H = K
            items.remove((x, y))
    print("Yes")

threading.Thread(target=main).start()