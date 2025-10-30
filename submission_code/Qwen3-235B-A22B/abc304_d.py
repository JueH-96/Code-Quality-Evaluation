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
    a_list = list(map(int, input[ptr:ptr + A]))
    ptr += A

    B = int(input[ptr])
    ptr += 1
    b_list = list(map(int, input[ptr:ptr + B]))
    ptr += B

    count = defaultdict(int)
    for p, q in strawberries:
        x = bisect.bisect_left(a_list, p)
        y = bisect.bisect_left(b_list, q)
        count[(x, y)] += 1

    total_cells = (A + 1) * (B + 1)
    count_cells = len(count)
    min_val = 0 if count_cells < total_cells else min(count.values())
    max_val = max(count.values())

    print(f"{min_val} {max_val}")

if __name__ == "__main__":
    main()