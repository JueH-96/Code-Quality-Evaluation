# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N,M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    total_steps = 0
    for i in range(N):
        A_i = A[i]
        B_i = B[i]
        if A_i == B_i:
            min_steps_i = 0
            total_steps += min_steps_i
            continue
        forbidden = set()
        if i > 0:
            forbidden.add(A[i-1])
        if i < N-1:
            forbidden.add(A[i+1])
        min_steps_i = float('inf')
        # Check clockwise direction
        D_xy_cw = (B_i - A_i) % M
        path_blocked = False
        for f in forbidden:
            D_xf_cw = (f - A_i) % M
            if D_xf_cw == 0:
                continue  # Forbidden node is at starting node, we can start from A_i != f
            if D_xf_cw < D_xy_cw:
                path_blocked = True
                break
        if not path_blocked:
            min_steps_i = D_xy_cw
        # Check counter-clockwise direction
        D_xy_ccw = (A_i - B_i) % M
        path_blocked = False
        for f in forbidden:
            D_xf_ccw = (A_i - f) % M
            if D_xf_ccw == 0:
                continue  # Forbidden node is at starting node, we can start from A_i != f
            if D_xf_ccw < D_xy_ccw:
                path_blocked = True
                break
        if not path_blocked:
            min_steps_i = min(min_steps_i, D_xy_ccw)
        if min_steps_i == float('inf'):
            print(-1)
            return
        total_steps += min_steps_i
    print(int(total_steps))

if __name__ == '__main__':
    threading.Thread(target=main).start()