# YOUR CODE HERE
import sys

import threading

def main():
    import sys
   
    import math
    sys.setrecursionlimit(1 << 25)

    N, M, L = map(int, sys.stdin.readline().split())
    a_list = list(map(int, sys.stdin.readline().split()))
    b_list = list(map(int, sys.stdin.readline().split()))
    forbidden_pairs = set()
    for _ in range(L):
        c_i, d_i = map(int, sys.stdin.readline().split())
        forbidden_pairs.add((c_i -1, d_i -1))

    # Sort a_list and b_list in descending order with their original indices
    a_with_indices = list(enumerate(a_list))
    b_with_indices = list(enumerate(b_list))
    a_with_indices.sort(key=lambda x: -x[1])
    b_with_indices.sort(key=lambda x: -x[1])

    K_a = min(1000, N)
    K_b = min(1000, M)

    max_sum = -1

    for i in range(K_a):
        idx_a, val_a = a_with_indices[i]
        for j in range(K_b):
            idx_b, val_b = b_with_indices[j]
            if (idx_a, idx_b) not in forbidden_pairs:
                current_sum = val_a + val_b
                if current_sum > max_sum:
                    max_sum = current_sum
                # Since lists are sorted descending, we can break early
                break  # No need to consider smaller b_list values for this idx_a
    print(max_sum)




threading.Thread(target=main).start()