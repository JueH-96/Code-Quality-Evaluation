# YOUR CODE HERE
import sys
import threading
from bisect import bisect_left, bisect_right

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    N = len(S)
    pre_one = [0]*(N+2)
    suff_two = [0]*(N+2)
    positions = []
    for i in range(1,N+1):
        pre_one[i] = pre_one[i-1] + (1 if S[i-1]=='1' else 0)
    for i in range(N, 0, -1):
        suff_two[i] = suff_two[i+1] + (1 if S[i-1]=='2' else 0)
    for i in range(1,N+1):
        if S[i-1]=='/':
            positions.append(i)
    # Read queries
    for _ in range(Q):
        s = sys.stdin.readline().strip()
        while s == '':
            s = sys.stdin.readline().strip()
        L, R = map(int, s.strip().split())
        idx_l = bisect_left(positions, L)
        idx_r = bisect_right(positions, R)
        max_L = 0
        for idx in positions[idx_l:idx_r]:
            n1 = pre_one[idx-1] - pre_one[L-1]
            n2 = suff_two[idx+1] - suff_two[R+1]
            n1 = max(0, n1)
            n2 = max(0, n2)
            L_val = 2 * min(n1, n2) +1
            max_possible_L = R - L +1
            if max_possible_L %2 == 0:
                max_possible_L -=1
            L_val = min(L_val, max_possible_L)
            max_L = max(max_L, L_val)
        print(max_L)

threading.Thread(target=main).start()