# YOUR CODE HERE
import sys

import math
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())

    N_original = N
    total = 0
    L = 1
    # For storing cumulative sums
    while True:
        if L == 1:
            palins = 10  # 0 to 9
        else:
            half_length = (L + 1) // 2
            palins = 9 * (10 ** (half_length - 1))
        total += palins
        if total >= N:
            break
        L += 1

    if L ==1:
        index_in_length = N
        palindrome = index_in_length -1  # Palindromes from 0 to 9
        print(palindrome)
        return
    else:
        prev_total = total - palins
        index_in_length = N - prev_total
        half_length = (L + 1) // 2
        start = 10 ** (half_length - 1)
        half_part = start + (index_in_length -1)
        half_str = str(half_part)
        if L % 2 == 0:
            palindrome_str = half_str + half_str[::-1]
        else:
            palindrome_str = half_str + half_str[-2::-1]
        print(palindrome_str)


if __name__ == "__main__":
    threading.Thread(target=main).start()