# YOUR CODE HERE
import sys

import threading
def main():
    import sys
    import math
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    K = list(map(int, N_and_rest[1:N+1]))
    total = sum(K)
    min_max = total  # Maximum possible sum

    from itertools import combinations

    for mask in range(1 << N):
        sum_A = 0
        sum_B = 0
        for i in range(N):
            if mask & (1 << i):
                sum_A += K[i]
            else:
                sum_B += K[i]
        curr_max = max(sum_A, sum_B)
        if curr_max < min_max:
            min_max = curr_max
    print(min_max)
threading.Thread(target=main).start()