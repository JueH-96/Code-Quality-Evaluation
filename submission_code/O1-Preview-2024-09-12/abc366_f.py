# YOUR CODE HERE

import sys
import threading
from math import log
from itertools import permutations

def main():
    import sys
    import math
    import random
    sys.setrecursionlimit(1 << 25)

    N, K = map(int, sys.stdin.readline().split())
    N = int(N)
    K = int(K)
    funcs = []
    for i in range(N):
        A_i, B_i = map(int, sys.stdin.readline().split())
        funcs.append((A_i + B_i, A_i, B_i, i))  # (Score, A_i, B_i, index)

    funcs.sort(reverse=True)
    K = min(K, N)
    top_funcs = funcs[:K]  # Take top K functions

    max_result = 0

    from itertools import permutations

    func_list = [(A_i, B_i) for (_, A_i, B_i, _) in top_funcs]
    indices = set()
    for _, _, _, idx in top_funcs:
        indices.add(idx)

    permuts = list(permutations(func_list))
    
    for perm in permuts:
        a = 1
        b = 0
        for A_i, B_i in perm:
            a_old = a
            a = a * A_i
            b = a_old * B_i + b
        result = a + b
        if result > max_result:
            max_result = result

    print(int(max_result))





threading.Thread(target=main).start()