# YOUR CODE HERE
import sys

import threading
def main():
    import sys
    import math

    N = int(sys.stdin.readline())
    min_deliciousness = {}

    for _ in range(N):
        A_i_str, C_i_str = sys.stdin.readline().split()
        A_i = int(A_i_str)
        C_i = int(C_i_str)
        if C_i not in min_deliciousness:
            min_deliciousness[C_i] = A_i
        else:
            min_deliciousness[C_i] = min(min_deliciousness[C_i], A_i)

    answer = max(min_deliciousness.values())
    print(answer)

threading.Thread(target=main).start()