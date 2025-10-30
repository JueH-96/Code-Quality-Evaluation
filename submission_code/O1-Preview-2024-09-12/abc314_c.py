# YOUR CODE HERE
import sys

import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))
    S_list = list(S)
    idx_list = {}
    for color in range(1, M+1):
        idx_list[color] = []
    for i, color in enumerate(C):
        idx_list[color].append(i+1)  # 1-based indices as per problem statement

    for color in range(1, M+1):
        indices = idx_list[color]
        if indices:
            chars = [S_list[i-1] for i in indices]
            shifted_chars = [chars[-1]] + chars[:-1]
            for idx, ch in zip(indices, shifted_chars):
                S_list[idx-1] = ch
    print(''.join(S_list))

threading.Thread(target=main).start()