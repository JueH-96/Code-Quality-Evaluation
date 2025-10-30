# YOUR CODE HERE
import sys

import threading

def main():
    N, *rest = list(map(int, sys.stdin.read().split()))
    A = rest[:N]

    dp_even = 0
    dp_odd = float('-inf')

    for A_i in A:
        dp_even_new = max(dp_even, dp_odd + 2 * A_i)
        dp_odd_new = max(dp_odd, dp_even + A_i)
        dp_even = dp_even_new
        dp_odd = dp_odd_new

    print(int(max(dp_even, dp_odd)))

threading.Thread(target=main).start()