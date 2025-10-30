# YOUR CODE HERE
import sys
import threading

def main():
    import math
    import bisect
    import sys
    sys.setrecursionlimit(1 << 25)
    T = int(sys.stdin.readline())
    for _ in range(T):
        N_str, X_str, K_str = sys.stdin.readline().split()
        N = int(N_str)
        X = int(X_str)
        K = int(K_str)
        h = X.bit_length()-1  # depth of X
        max_depth = N.bit_length() - 1
        ans = 0
        max_possible_depth = max_depth
        for d_u in range(max_possible_depth+1):
            temp = h + d_u - K
            if temp % 2 != 0:
                continue
            l = temp // 2
            if l < 0 or l > min(h, d_u):
                continue
            if l == 0:
                prefix_bits = 0
            else:
                prefix_bits = X >> (h - l)
            min_label = (prefix_bits) << (d_u - l)
            max_label = ((prefix_bits + 1) << (d_u - l)) - 1
            l_d = 1 << d_u
            r_d = min((1 << (d_u + 1)) -1, N)
            min_label = max(min_label, l_d)
            max_label = min(max_label, r_d)
            if min_label > max_label:
                continue
            nodes = max_label - min_label +1
            ans += nodes
        print(ans)