# YOUR CODE HERE
import sys
import math
from collections import defaultdict

def main():
    import sys
    import math
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    counts = defaultdict(int)
    max_j = 0
    for a in A:
        if a >0:
            max_j = max(max_j, a.bit_length())
    # Since 2^100 is 1.2676506e+30, j can go up to 100
    for j in range(0, 101):
        p = 1 << j
        for a in A:
            if p > a:
                i = p - a
                counts[i] +=1
    if counts:
        print(max(counts.values()))
    else:
        print(0)

if __name__ == "__main__":
    main()