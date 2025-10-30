# YOUR CODE HERE
import sys
import bisect

import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)

    N_T = sys.stdin.readline().strip()
    while N_T == '':
        N_T = sys.stdin.readline().strip()
    N, T = map(int, N_T.strip().split())
    N = int(N)
    T = int(T)

    S_line = sys.stdin.readline().strip()
    while len(S_line) < N:
        S_line += sys.stdin.readline().strip()
    S = S_line.strip()

    pos_line = ''
    while len(pos_line.strip().split()) < N:
        pos_line += sys.stdin.readline()
    xi_list = list(map(int, pos_line.strip().split()))
    xi10_list = [xi * 10 for xi in xi_list]

    T10 = T * 10 + 1
    D = 2 * T10

    positions_left = []
    for di, xi10 in zip(S, xi10_list):
        if di == '0':
            # direction -1
            positions_left.append(xi10)

    positions_left.sort()

    total = 0
    for di, xi10 in zip(S, xi10_list):
        if di == '1':
            # direction +1
            left_index = bisect.bisect_left(positions_left, xi10)
            right_index = bisect.bisect_right(positions_left, xi10 + D)
            count = right_index - left_index
            total += count

    print(total)

threading.Thread(target=main).start()