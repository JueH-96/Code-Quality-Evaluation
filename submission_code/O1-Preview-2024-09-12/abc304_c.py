# YOUR CODE HERE
import sys
import threading

def main():
    import math
    N, D = map(int, sys.stdin.readline().split())
    coords = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    D_squared = D * D
    infected = [False] * N
    infected[0] = True
    queue = [0]

    while queue:
        i = queue.pop(0)
        xi, yi = coords[i]
        for j in range(N):
            if not infected[j]:
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist_squared = dx * dx + dy * dy
                if dist_squared <= D_squared:
                    infected[j] = True
                    queue.append(j)

    for inf in infected:
        print('Yes' if inf else 'No')

if __name__ == '__main__':
    threading.Thread(target=main).start()