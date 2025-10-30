import bisect
from collections import defaultdict

def main():
    import sys
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
    a = list(map(int, input[ptr:ptr + A]))
    ptr += A

    B = int(input[ptr])
    ptr += 1
    b = list(map(int, input[ptr:ptr + B]))
    ptr += B

    counter = defaultdict(int)

    for p, q in strawberries:
        x_idx = bisect.bisect_left(a, p)
        y_idx = bisect.bisect_left(b, q)
        counter[(x_idx, y_idx)] += 1

    max_count = max(counter.values())

    total_cells = (A + 1) * (B + 1)
    unique_cells = len(counter)

    if unique_cells < total_cells:
        min_count = 0
    else:
        min_count = min(counter.values())

    print(f"{min_count} {max_count}")

if __name__ == "__main__":
    main()