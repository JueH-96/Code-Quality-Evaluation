import bisect
from collections import defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    W = int(input[ptr])
    ptr += 1
    H = int(input[ptr])
    ptr += 1
    N = int(input[ptr])
    ptr += 1
    strawberries = []
    for _ in range(N):
        p = int(input[ptr])
        ptr += 1
        q = int(input[ptr])
        ptr += 1
        strawberries.append((p, q))
    A = int(input[ptr])
    ptr += 1
    a_list = list(map(int, input[ptr:ptr+A]))
    ptr += A
    B = int(input[ptr])
    ptr += 1
    b_list = list(map(int, input[ptr:ptr+B]))
    ptr += B

    counts = defaultdict(int)
    for p, q in strawberries:
        v = bisect.bisect_right(a_list, p)
        h = bisect.bisect_right(b_list, q)
        counts[(v, h)] += 1

    total_pieces = (A + 1) * (B + 1)
    non_empty = len(counts)
    max_val = max(counts.values()) if counts else 0
    if non_empty < total_pieces:
        min_val = 0
    else:
        min_val = min(counts.values())

    print(min_val, max_val)

if __name__ == '__main__':
    main()