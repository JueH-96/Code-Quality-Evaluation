# YOUR CODE HERE
import sys
import math
from collections import Counter

import threading
def main():
    import sys

    input_lines = sys.stdin.read().split()
    N = int(input_lines[0])
    S = input_lines[1]
    s_count = Counter(S)
    ans = set()

    max_sqrt = int(math.isqrt(10**N -1))
    for i in range(1, max_sqrt +1):
        square_num = i * i
        square_num_str = str(square_num).zfill(N)
        if len(square_num_str) > N:
            continue
        square_count = Counter(square_num_str)
        if square_count == s_count:
            ans.add(square_num)
    print(len(ans))

threading.Thread(target=main).start()